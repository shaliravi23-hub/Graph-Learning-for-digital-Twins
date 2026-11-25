import argparse
import ipaddress
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from scapy.utils import PcapReader, PcapNgReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


def is_private_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False


def is_multicast_or_broadcast(ip_str: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_multicast:
            return True
        if ip_str == "255.255.255.255":
            return True
        return False
    except ValueError:
        return True


def open_packet_reader(file_path: Path):
    # Some files are named .pcap but are actually pcapng.
    try:
        reader = PcapNgReader(str(file_path))
        _ = next(iter(reader))
        reader.close()
        return PcapNgReader(str(file_path))
    except Exception:
        try:
            return PcapReader(str(file_path))
        except Exception as e:
            raise RuntimeError(f"Could not open {file_path}: {e}")


def get_packet_fields(pkt):
    ts = float(pkt.time)
    pkt_len = int(len(pkt))

    src_ip = None
    dst_ip = None
    proto = "OTHER"
    sport = 0
    dport = 0

    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto_num = int(pkt[IP].proto)
        if proto_num == 6:
            proto = "TCP"
        elif proto_num == 17:
            proto = "UDP"
        elif proto_num == 1:
            proto = "ICMP"
    elif IPv6 in pkt:
        src_ip = pkt[IPv6].src
        dst_ip = pkt[IPv6].dst
        proto_num = int(pkt[IPv6].nh)
        if proto_num == 6:
            proto = "TCP"
        elif proto_num == 17:
            proto = "UDP"
        elif proto_num == 58:
            proto = "ICMPv6"
    else:
        return None

    if TCP in pkt:
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
    elif UDP in pkt:
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)

    return {
        "timestamp": ts,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "proto": proto,
        "sport": sport,
        "dport": dport,
        "pkt_len": pkt_len,
    }


def new_edge_stats():
    return {
        "packet_count": 0,
        "byte_count": 0,
        "tcp_count": 0,
        "udp_count": 0,
        "icmp_count": 0,
        "first_ts": None,
        "last_ts": None,
        "flow_keys": set(),
    }


def update_edge_stats(stats, ts, pkt_len, proto, flow_key):
    stats["packet_count"] += 1
    stats["byte_count"] += pkt_len

    if proto == "TCP":
        stats["tcp_count"] += 1
    elif proto == "UDP":
        stats["udp_count"] += 1
    elif proto in {"ICMP", "ICMPv6"}:
        stats["icmp_count"] += 1

    if stats["first_ts"] is None or ts < stats["first_ts"]:
        stats["first_ts"] = ts
    if stats["last_ts"] is None or ts > stats["last_ts"]:
        stats["last_ts"] = ts

    stats["flow_keys"].add(flow_key)


def finalize_flow(flow_key, flow_data, output_rows):
    duration = max(0.0, flow_data["last_ts"] - flow_data["start_ts"])
    total_packets = flow_data["packet_count"]
    total_bytes = flow_data["byte_count"]
    mean_packet_size = total_bytes / total_packets if total_packets > 0 else 0.0

    output_rows.append({
        "src_ip": flow_key[0],
        "dst_ip": flow_key[1],
        "sport": flow_key[2],
        "dport": flow_key[3],
        "proto": flow_key[4],
        "start_ts": flow_data["start_ts"],
        "end_ts": flow_data["last_ts"],
        "duration": duration,
        "packet_count": total_packets,
        "byte_count": total_bytes,
        "mean_packet_size": mean_packet_size,
    })


def process_one_pcap(file_path: Path, output_dir: Path, window_sizes, idle_timeout: int):
    date_id = file_path.stem
    flows_dir = output_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)

    graph_dirs = {}
    for w in window_sizes:
        gdir = output_dir / f"graphs_{w}s" / date_id
        gdir.mkdir(parents=True, exist_ok=True)
        graph_dirs[w] = gdir

    reader = open_packet_reader(file_path)

    active_flows = {}
    finalized_flows = []

    # Per-window graph aggregation
    # window_edge_aggs[window_size][window_id][(src, dst)] = stats
    window_edge_aggs = {
        w: defaultdict(lambda: defaultdict(new_edge_stats))
        for w in window_sizes
    }

    first_ts = None
    packet_counter = 0

    stats_counter = {
        "total_packets": 0,
        "ipv4_ipv6_packets": 0,
        "internal_internal_packets": 0,
        "internal_external_packets": 0,
        "external_external_packets": 0,
        "multicast_broadcast_packets": 0,
        "tcp_packets": 0,
        "udp_packets": 0,
        "icmp_packets": 0,
        "unique_internal_nodes": set(),
        "unique_external_nodes": set(),
    }

    for pkt in tqdm(reader, desc=f"Reading {file_path.name}"):
        parsed = get_packet_fields(pkt)
        if parsed is None:
            continue

        ts = parsed["timestamp"]
        src_ip = parsed["src_ip"]
        dst_ip = parsed["dst_ip"]
        proto = parsed["proto"]
        sport = parsed["sport"]
        dport = parsed["dport"]
        pkt_len = parsed["pkt_len"]

        if first_ts is None:
            first_ts = ts

        stats_counter["total_packets"] += 1
        stats_counter["ipv4_ipv6_packets"] += 1
        packet_counter += 1

        if proto == "TCP":
            stats_counter["tcp_packets"] += 1
        elif proto == "UDP":
            stats_counter["udp_packets"] += 1
        elif proto in {"ICMP", "ICMPv6"}:
            stats_counter["icmp_packets"] += 1

        src_internal = is_private_ip(src_ip)
        dst_internal = is_private_ip(dst_ip)

        if is_multicast_or_broadcast(src_ip) or is_multicast_or_broadcast(dst_ip):
            stats_counter["multicast_broadcast_packets"] += 1
            continue

        if src_internal:
            stats_counter["unique_internal_nodes"].add(src_ip)
        else:
            stats_counter["unique_external_nodes"].add(src_ip)

        if dst_internal:
            stats_counter["unique_internal_nodes"].add(dst_ip)
        else:
            stats_counter["unique_external_nodes"].add(dst_ip)

        if src_internal and dst_internal:
            stats_counter["internal_internal_packets"] += 1
        elif src_internal != dst_internal:
            stats_counter["internal_external_packets"] += 1
        else:
            stats_counter["external_external_packets"] += 1

        # ---------- Flow extraction ----------
        flow_key = (src_ip, dst_ip, sport, dport, proto)
        if flow_key in active_flows:
            if ts - active_flows[flow_key]["last_ts"] > idle_timeout:
                finalize_flow(flow_key, active_flows[flow_key], finalized_flows)
                active_flows[flow_key] = {
                    "start_ts": ts,
                    "last_ts": ts,
                    "packet_count": 1,
                    "byte_count": pkt_len,
                }
            else:
                active_flows[flow_key]["last_ts"] = ts
                active_flows[flow_key]["packet_count"] += 1
                active_flows[flow_key]["byte_count"] += pkt_len
        else:
            active_flows[flow_key] = {
                "start_ts": ts,
                "last_ts": ts,
                "packet_count": 1,
                "byte_count": pkt_len,
            }

        # ---------- Main graph construction ----------
        # Keep only internal-to-internal traffic for the main paper graph.
        if not (src_internal and dst_internal):
            continue

        for w in window_sizes:
            window_id = int((ts - first_ts) // w)
            edge_stats = window_edge_aggs[w][window_id][(src_ip, dst_ip)]
            update_edge_stats(edge_stats, ts, pkt_len, proto, flow_key)

    # Finalize remaining flows
    for fk, fd in active_flows.items():
        finalize_flow(fk, fd, finalized_flows)

    reader.close()

    # Save flows
    flows_df = pd.DataFrame(finalized_flows)
    flows_df.sort_values(["start_ts", "src_ip", "dst_ip"], inplace=True, ignore_index=True)
    flows_df.to_csv(flows_dir / f"flows_{date_id}.csv", index=False)

    # Save graphs, node/edge tables, and stats for each window size
    summary_by_window = {}

    for w in window_sizes:
        graph_records = []
        edge_rows = []
        node_rows = []
        graph_snapshots = []

        for window_id in sorted(window_edge_aggs[w].keys()):
            edge_dict = window_edge_aggs[w][window_id]
            G = nx.DiGraph()

            # First pass: add edges
            for (src, dst), s in edge_dict.items():
                packet_count = s["packet_count"]
                byte_count = s["byte_count"]
                mean_packet_size = byte_count / packet_count if packet_count > 0 else 0.0
                duration = max(0.0, (s["last_ts"] - s["first_ts"])) if s["first_ts"] is not None else 0.0
                flow_count = len(s["flow_keys"])

                reverse = edge_dict.get((dst, src), None)
                reverse_bytes = reverse["byte_count"] if reverse is not None else 0
                bidir_byte_ratio = byte_count / (reverse_bytes + 1e-9)

                G.add_edge(
                    src,
                    dst,
                    packet_count=packet_count,
                    byte_count=byte_count,
                    flow_count=flow_count,
                    mean_packet_size=mean_packet_size,
                    duration=duration,
                    tcp_ratio=s["tcp_count"] / packet_count if packet_count else 0.0,
                    udp_ratio=s["udp_count"] / packet_count if packet_count else 0.0,
                    icmp_ratio=s["icmp_count"] / packet_count if packet_count else 0.0,
                    bidir_byte_ratio=bidir_byte_ratio,
                )

                edge_rows.append({
                    "window_id": window_id,
                    "src_ip": src,
                    "dst_ip": dst,
                    "packet_count": packet_count,
                    "byte_count": byte_count,
                    "flow_count": flow_count,
                    "mean_packet_size": mean_packet_size,
                    "duration": duration,
                    "tcp_ratio": s["tcp_count"] / packet_count if packet_count else 0.0,
                    "udp_ratio": s["udp_count"] / packet_count if packet_count else 0.0,
                    "icmp_ratio": s["icmp_count"] / packet_count if packet_count else 0.0,
                    "bidir_byte_ratio": bidir_byte_ratio,
                })

            # Second pass: node features
            for node in G.nodes():
                in_edges = list(G.in_edges(node, data=True))
                out_edges = list(G.out_edges(node, data=True))

                in_degree = len(in_edges)
                out_degree = len(out_edges)

                in_bytes = sum(d["byte_count"] for _, _, d in in_edges)
                out_bytes = sum(d["byte_count"] for _, _, d in out_edges)

                in_packets = sum(d["packet_count"] for _, _, d in in_edges)
                out_packets = sum(d["packet_count"] for _, _, d in out_edges)

                active_edge_count = in_degree + out_degree
                total_packets = in_packets + out_packets

                tcp_packets = sum(d["tcp_ratio"] * d["packet_count"] for _, _, d in in_edges + out_edges)
                udp_packets = sum(d["udp_ratio"] * d["packet_count"] for _, _, d in in_edges + out_edges)
                icmp_packets = sum(d["icmp_ratio"] * d["packet_count"] for _, _, d in in_edges + out_edges)

                G.nodes[node]["in_degree"] = in_degree
                G.nodes[node]["out_degree"] = out_degree
                G.nodes[node]["in_bytes"] = in_bytes
                G.nodes[node]["out_bytes"] = out_bytes
                G.nodes[node]["in_packets"] = in_packets
                G.nodes[node]["out_packets"] = out_packets
                G.nodes[node]["active_edge_count"] = active_edge_count
                G.nodes[node]["tcp_ratio"] = tcp_packets / total_packets if total_packets else 0.0
                G.nodes[node]["udp_ratio"] = udp_packets / total_packets if total_packets else 0.0
                G.nodes[node]["icmp_ratio"] = icmp_packets / total_packets if total_packets else 0.0

                node_rows.append({
                    "window_id": window_id,
                    "node_ip": node,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                    "in_packets": in_packets,
                    "out_packets": out_packets,
                    "active_edge_count": active_edge_count,
                    "tcp_ratio": tcp_packets / total_packets if total_packets else 0.0,
                    "udp_ratio": udp_packets / total_packets if total_packets else 0.0,
                    "icmp_ratio": icmp_packets / total_packets if total_packets else 0.0,
                })

            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            density = nx.density(G) if num_nodes > 1 else 0.0
            weak_components = nx.number_weakly_connected_components(G) if num_nodes > 0 else 0

            graph_records.append({
                "window_id": window_id,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "weak_components": weak_components,
            })

            graph_snapshots.append((window_id, G))

        edges_df = pd.DataFrame(edge_rows)
        nodes_df = pd.DataFrame(node_rows)
        graph_df = pd.DataFrame(graph_records)

        gdir = graph_dirs[w]
        edges_df.to_csv(gdir / "edges.csv", index=False)
        nodes_df.to_csv(gdir / "nodes.csv", index=False)
        graph_df.to_csv(gdir / "graph_stats.csv", index=False)

        with open(gdir / "graph_snapshots.pkl", "wb") as f:
            pickle.dump(graph_snapshots, f)

        summary_by_window[w] = {
            "num_windows": int(graph_df.shape[0]),
            "avg_nodes_per_window": float(graph_df["num_nodes"].mean()) if not graph_df.empty else 0.0,
            "avg_edges_per_window": float(graph_df["num_edges"].mean()) if not graph_df.empty else 0.0,
            "avg_density": float(graph_df["density"].mean()) if not graph_df.empty else 0.0,
        }

    final_stats = {
        "date_id": date_id,
        "pcap_file": str(file_path),
        "packet_count_used_for_ip_analysis": stats_counter["ipv4_ipv6_packets"],
        "total_packets_seen": stats_counter["total_packets"],
        "internal_internal_packets": stats_counter["internal_internal_packets"],
        "internal_external_packets": stats_counter["internal_external_packets"],
        "external_external_packets": stats_counter["external_external_packets"],
        "multicast_broadcast_packets": stats_counter["multicast_broadcast_packets"],
        "tcp_packets": stats_counter["tcp_packets"],
        "udp_packets": stats_counter["udp_packets"],
        "icmp_packets": stats_counter["icmp_packets"],
        "num_unique_internal_nodes": len(stats_counter["unique_internal_nodes"]),
        "num_unique_external_nodes": len(stats_counter["unique_external_nodes"]),
        "windows": summary_by_window,
    }

    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / f"stats_{date_id}.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    print(f"[DONE] {date_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap_dir", type=str, required=True, help="Folder containing PCAP/PCAPNG files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--windows", type=int, nargs="+", default=[300, 60], help="Window sizes in seconds")
    parser.add_argument("--idle_timeout", type=int, default=60, help="Idle timeout for flow extraction")
    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in pcap_dir.iterdir() if p.suffix.lower() in {".pcap", ".pcapng"}]
    )

    if not files:
        raise FileNotFoundError(f"No PCAP/PCAPNG files found in {pcap_dir}")

    for file_path in files:
        process_one_pcap(
            file_path=file_path,
            output_dir=output_dir,
            window_sizes=args.windows,
            idle_timeout=args.idle_timeout,
        )

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()