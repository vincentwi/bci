#!/bin/bash
# Download ALL BCI speech data from OSF (https://osf.io/49rt7/)
set -e
BASE="/mnt/home/vincent.wilmet/data"

# Create all directories
for d in train/2022_09_23 train/2022_09_30 train/2022_10_05 train/2022_10_06 \
         train/2022_10_10 train/2022_10_27 \
         test/2022_11_03 validation/2022_11_04 \
         online_sessions/2023_04_14 online_sessions/2023_04_18 online_sessions/2023_04_21 \
         syllable/2022_09_23 syllable/2022_09_28 syllable/2022_09_30 \
         syllable/2022_10_05 syllable/2022_10_10 syllable/2022_10_27 \
         syllable/2022_11_03 syllable/2022_11_04 syllable/2023_04_14; do
    mkdir -p "$BASE/$d"
done

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null)" -gt 100 ]; then
        echo "  SKIP (exists): $(basename "$dest")"
        return
    fi
    wget -q -c -O "$dest" "$url" && echo "  OK: $(basename "$dest")" || echo "  FAIL: $(basename "$dest")"
}

echo "=== KeywordReading/train/2022_09_23 ==="
download "https://osf.io/download/zvub9/" "$BASE/train/2022_09_23/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/66295a07c5851a1391f66fb8/" "$BASE/train/2022_09_23/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/66295a05d8960717b61b1594/" "$BASE/train/2022_09_23/KeywordReading_Overt_R01_trials.lab" &
download "https://osf.io/download/4mvse/" "$BASE/train/2022_09_23/KeywordReading_Overt_R02.mat" &
download "https://osf.io/download/66295a0980d25c1396f91aa3/" "$BASE/train/2022_09_23/KeywordReading_Overt_R02.wav" &
download "https://osf.io/download/s3cq6/" "$BASE/train/2022_09_23/KeywordReading_Overt_R02_trials.lab" &
download "https://osf.io/download/66295a103145c6120b6a83c6/" "$BASE/train/2022_09_23/KeywordReading_Overt_R03.mat" &
download "https://osf.io/download/66295a0ec5851a1396f670ce/" "$BASE/train/2022_09_23/KeywordReading_Overt_R03.wav" &
download "https://osf.io/download/66295a0f80d25c1392f919d3/" "$BASE/train/2022_09_23/KeywordReading_Overt_R03_trials.lab" &
download "https://osf.io/download/66295a0f3145c6120b6a83c4/" "$BASE/train/2022_09_23/KeywordReading_Overt_R04.mat" &
download "https://osf.io/download/66295a11c5851a1391f66fbd/" "$BASE/train/2022_09_23/KeywordReading_Overt_R04.wav" &
download "https://osf.io/download/5vj6f/" "$BASE/train/2022_09_23/KeywordReading_Overt_R04_trials.lab" &
wait

echo "=== KeywordReading/train/2022_09_30 ==="
download "https://osf.io/download/66295a9380d25c1396f91b48/" "$BASE/train/2022_09_30/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/66295a78c5851a139ff66f46/" "$BASE/train/2022_09_30/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/66295a793145c612076a831a/" "$BASE/train/2022_09_30/KeywordReading_Overt_R01_trials.lab" &
download "https://osf.io/download/66295a9380d25c1396f91b48/" "$BASE/train/2022_09_30/KeywordReading_Overt_R02.mat" &
download "https://osf.io/download/66295a8580d25c1396f91b28/" "$BASE/train/2022_09_30/KeywordReading_Overt_R02.wav" &
download "https://osf.io/download/66295a7ef49cdf17cfd2557c/" "$BASE/train/2022_09_30/KeywordReading_Overt_R02_trials.lab" &
download "https://osf.io/download/66295a93d8960717b31b1469/" "$BASE/train/2022_09_30/KeywordReading_Overt_R03.mat" &
download "https://osf.io/download/66295a863145c6120b6a8438/" "$BASE/train/2022_09_30/KeywordReading_Overt_R03.wav" &
download "https://osf.io/download/66295a8ac5851a139ff66f4a/" "$BASE/train/2022_09_30/KeywordReading_Overt_R03_trials.lab" &
download "https://osf.io/download/66295a9580d25c1395f91a98/" "$BASE/train/2022_09_30/KeywordReading_Overt_R04.mat" &
download "https://osf.io/download/66295a92c5851a139ff66f4e/" "$BASE/train/2022_09_30/KeywordReading_Overt_R04.wav" &
download "https://osf.io/download/66295a96d8960717b21b1507/" "$BASE/train/2022_09_30/KeywordReading_Overt_R04_trials.lab" &
wait

echo "=== KeywordReading/train/2022_10_05 ==="
download "https://osf.io/download/66295ac53145c6120b6a8490/" "$BASE/train/2022_10_05/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/66295abfd8960717b61b163c/" "$BASE/train/2022_10_05/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/66295abe80d25c1395f91abe/" "$BASE/train/2022_10_05/KeywordReading_Overt_R01_trials.lab" &
download "https://osf.io/download/66295acbd8960717b61b164a/" "$BASE/train/2022_10_05/KeywordReading_Overt_R02.mat" &
download "https://osf.io/download/66295ac480d25c1392f91a10/" "$BASE/train/2022_10_05/KeywordReading_Overt_R02.wav" &
download "https://osf.io/download/66295ac780d25c1396f91b86/" "$BASE/train/2022_10_05/KeywordReading_Overt_R02_trials.lab" &
download "https://osf.io/download/66295ac980d25c1396f91b8a/" "$BASE/train/2022_10_05/KeywordReading_Overt_R04.mat" &
download "https://osf.io/download/66295acf3145c6120b6a8499/" "$BASE/train/2022_10_05/KeywordReading_Overt_R04.wav" &
download "https://osf.io/download/66295acfc5851a139ff66f88/" "$BASE/train/2022_10_05/KeywordReading_Overt_R04_trials.lab" &
wait

echo "=== KeywordReading/train/2022_10_06 ==="
download "https://osf.io/download/66295aed3145c612026a8293/" "$BASE/train/2022_10_06/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/66295aeed8960717ac1b146f/" "$BASE/train/2022_10_06/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/66295aeed8960717b71b17af/" "$BASE/train/2022_10_06/KeywordReading_Overt_R01_trials.lab" &
wait

echo "=== KeywordReading/train/2022_10_10 ==="
download "https://osf.io/download/s9ue2/" "$BASE/train/2022_10_10/KeywordReading_Overt_R02.mat" &
download "https://osf.io/download/66295b0ac5851a13b3f66f05/" "$BASE/train/2022_10_10/KeywordReading_Overt_R02.wav" &
download "https://osf.io/download/4u9f5/" "$BASE/train/2022_10_10/KeywordReading_Overt_R02_trials.lab" &
wait

echo "=== KeywordReading/train/2022_10_27 ==="
download "https://osf.io/download/2by3e/" "$BASE/train/2022_10_27/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/66295b34f49cdf17ead25273/" "$BASE/train/2022_10_27/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/66295b32f49cdf17efd2527a/" "$BASE/train/2022_10_27/KeywordReading_Overt_R01_trials.lab" &
wait

echo "=== KeywordReading/test/2022_11_03 ==="
download "https://osf.io/download/mx28j/" "$BASE/test/2022_11_03/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/km39q/" "$BASE/test/2022_11_03/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/7ap5b/" "$BASE/test/2022_11_03/KeywordReading_Overt_R01_trials.lab" &
wait

echo "=== KeywordReading/validation/2022_11_04 ==="
download "https://osf.io/download/692mp/" "$BASE/validation/2022_11_04/KeywordReading_Overt_R01.mat" &
download "https://osf.io/download/hftk3/" "$BASE/validation/2022_11_04/KeywordReading_Overt_R01.wav" &
download "https://osf.io/download/raxhm/" "$BASE/validation/2022_11_04/KeywordReading_Overt_R01_trials.lab" &
wait

echo "=== KeywordReading/online_sessions ==="
download "https://osf.io/download/qhzg7/" "$BASE/online_sessions/2023_04_14/KeywordSynthesis_Overt_R01.mat" &
download "https://osf.io/download/vs728/" "$BASE/online_sessions/2023_04_14/KeywordSynthesis_Overt_R01.wav" &
download "https://osf.io/download/sfh5d/" "$BASE/online_sessions/2023_04_14/KeywordSynthesis_Overt_R01_trials.lab" &
download "https://osf.io/download/u4wes/" "$BASE/online_sessions/2023_04_18/KeywordSynthesis_Overt_R01.mat" &
download "https://osf.io/download/xfrew/" "$BASE/online_sessions/2023_04_18/KeywordSynthesis_Overt_R01.wav" &
download "https://osf.io/download/6629596680d25c1391f91951/" "$BASE/online_sessions/2023_04_18/KeywordSynthesis_Overt_R01_trials.lab" &
download "https://osf.io/download/6629598b80d25c1396f91a48/" "$BASE/online_sessions/2023_04_21/KeywordSynthesis_Overt_R01.mat" &
download "https://osf.io/download/yt7s8/" "$BASE/online_sessions/2023_04_21/KeywordSynthesis_Overt_R01.wav" &
download "https://osf.io/download/66295989d8960717ac1b1444/" "$BASE/online_sessions/2023_04_21/KeywordSynthesis_Overt_R01_trials.lab" &
wait

echo "=== SyllableRepetition (all remaining sessions) ==="
download "https://osf.io/download/e92da/" "$BASE/syllable/2022_09_23/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/sfm3j/" "$BASE/syllable/2022_09_28/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/syvr6/" "$BASE/syllable/2022_09_30/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/wpszt/" "$BASE/syllable/2022_10_05/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/dsrej/" "$BASE/syllable/2022_10_10/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/662958ac80d25c1391f91941/" "$BASE/syllable/2022_10_27/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/662958c9d8960717b21b1463/" "$BASE/syllable/2022_11_03/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/qf6uk/" "$BASE/syllable/2022_11_04/SyllableRepetition_Overt.mat" &
download "https://osf.io/download/66295902c5851a1396f66f92/" "$BASE/syllable/2023_04_14/SyllableRepetition_Overt.mat" &
wait

echo ""
echo "=== Download Complete ==="
echo "Train sessions:"
for d in "$BASE"/train/*/; do echo "  $(basename $d): $(ls "$d"/*.mat 2>/dev/null | wc -l) runs"; done
echo "Test sessions:"
for d in "$BASE"/test/*/; do echo "  $(basename $d): $(ls "$d"/*.mat 2>/dev/null | wc -l) runs"; done
echo "Validation sessions:"
for d in "$BASE"/validation/*/; do echo "  $(basename $d): $(ls "$d"/*.mat 2>/dev/null | wc -l) runs"; done
echo "Online sessions:"
for d in "$BASE"/online_sessions/*/; do echo "  $(basename $d): $(ls "$d"/*.mat 2>/dev/null | wc -l) runs"; done
echo "Syllable sessions:"
for d in "$BASE"/syllable/*/; do echo "  $(basename $d): $(ls "$d"/*.mat 2>/dev/null | wc -l) files"; done
