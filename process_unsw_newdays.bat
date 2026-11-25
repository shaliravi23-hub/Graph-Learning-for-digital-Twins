@echo off
REM Run from D:\twin in Anaconda Prompt (base). Safe to rerun: finished days are skipped.
REM Step 1 re-processes one day we already used (9 Oct) and checks it reproduces the existing graphs exactly.
REM Step 2 processes the 12 new UNSW days one at a time (unpack, 300 s graphs, delete the unpacked capture).
REM The downloaded .tar.gz files are kept. Needs about 13 GB free on D: while the largest day (12 Oct) is unpacked.
set UNSW_SRC=D:\unsw
set PCAP_TMP=D:\unsw\pcap_tmp
set UNSW_OUT=D:\unsw\outputs
if not exist %PCAP_TMP% mkdir %PCAP_TMP%
if not exist %UNSW_SRC%\check\16-10-09.tar.gz (
  echo Download 16-10-09.tar.gz into D:\unsw\check first.
  exit /b 1
)
if not exist %UNSW_SRC%\check_out\graphs_300s\16-10-09\graph_snapshots.pkl (
  pushd %PCAP_TMP%
  tar -xzf %UNSW_SRC%\check\16-10-09.tar.gz || (popd & exit /b 1)
  popd
  python preprocess_pcaps_clean.py --pcap_dir %PCAP_TMP% --output_dir %UNSW_SRC%\check_out --windows 300 || exit /b 1
  del %PCAP_TMP%\16-10-09.pcap
)
python check_reprocess.py --new %UNSW_SRC%\check_out\graphs_300s\16-10-09 --old outputs\graphs_300s\16-10-09 || exit /b 1
for %%d in (16-09-28 16-09-29 16-09-30 16-10-01 16-10-02 16-10-03 16-10-04 16-10-05 16-10-06 16-10-07 16-10-11 16-10-12) do (
  if not exist %UNSW_OUT%\graphs_300s\%%d\graph_snapshots.pkl (
    if exist %PCAP_TMP%\*.pcap del %PCAP_TMP%\*.pcap
    pushd %PCAP_TMP%
    tar -xzf %UNSW_SRC%\%%d.tar.gz || (popd & exit /b 1)
    popd
    python preprocess_pcaps_clean.py --pcap_dir %PCAP_TMP% --output_dir %UNSW_OUT% --windows 300 || exit /b 1
    del %PCAP_TMP%\%%d.pcap
  )
)
echo All 12 new UNSW days processed.
