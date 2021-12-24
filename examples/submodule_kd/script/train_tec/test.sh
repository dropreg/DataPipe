

cat en.pred0 | awk -F '\t' '{print $NF}' | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > en.final.pred
cat en.ref | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > en.final.ref
fairseq-score --sys en.final.pred --ref en.final.ref
