categories=('' 'acpa' 'concdel' 'danais' 'dcppc' 'doris' 'styx')
for catDmd in "${categories[@]}"
do
    echo "Category: $catDmd"
    bash compare.sh acc $catDmd
    bash compare.sh f1-macro-avg $catDmd
    bash compare.sh balanced-acc $catDmd
done
