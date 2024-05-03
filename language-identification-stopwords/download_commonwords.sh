mkdir -p commonwords && cd commonwords

for lang_code in af az bg cs da de el en es fi fr hr it ko nl no pl ru ur; do
    wget -O commonwords-$lang_code.txt https://raw.githubusercontent.com/frekwencja/most-common-words-multilingual/main/data/wordfrequency.info/$lang_code.txt
done

wget -O commonwords-zh_raw.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/zh/zh_50k.txt
awk '{print $1}' commonwords-zh_raw.txt > commonwords-zh.txt
rm commonwords-zh_raw.txt