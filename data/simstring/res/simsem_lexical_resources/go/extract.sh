#!/bin/sh

MYSQL_CMD='mysql -u root -p go'

# Biological Process
cat acc_and_name.sql | sed -e 's|%%TERM_NAME%%|biological_process|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_name_biological_process.list
cat acc_and_name_biological_process.list | ./variants.py | sort -n | uniq \
    > acc_and_name_biological_process_variants.list
cat acc_and_exact_synonyms.sql | sed -e 's|%%TERM_NAME%%|biological_process|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_exact_synonyms_biological_process.list
cat acc_and_exact_synonyms_biological_process.list | ./variants.py | sort -n | uniq \
    > acc_and_exact_synonyms_biological_process_variants.list

    
# Cellular Component
cat acc_and_name.sql | sed -e 's|%%TERM_NAME%%|cellular_component|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_name_cellular_component.list
cat acc_and_name_cellular_component.list | ./variants.py | sort -n | uniq \
    > acc_and_name_cellular_component_variants.list
cat acc_and_exact_synonyms.sql | sed -e 's|%%TERM_NAME%%|cellular_component|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_exact_synonyms_cellular_component.list
cat acc_and_exact_synonyms_cellular_component.list | ./variants.py | sort -n | uniq \
    > acc_and_exact_synonyms_cellular_component_variants.list
    
# Molecular Function
cat acc_and_name.sql | sed -e 's|%%TERM_NAME%%|molecular_function|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_name_molecular_function.list
cat acc_and_name_molecular_function.list | ./variants.py | sort -n | uniq \
    > acc_and_name_molecular_function_variants.list
cat acc_and_exact_synonyms.sql | sed -e 's|%%TERM_NAME%%|molecular_function|g' \
    | ${MYSQL_CMD} | sed '1d' | sort -n | uniq > acc_and_exact_synonyms_molecular_function.list
cat acc_and_exact_synonyms_molecular_function.list | ./variants.py | sort -n | uniq \
    > acc_and_exact_synonyms_molecular_function_variants.list

# Convert all lists into token listings
for LISTING in `ls *.list`
do
    cat ${LISTING} | cut -f 2 | sort | uniq > ${LISTING}.tokens
done
