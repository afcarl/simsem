SELECT DISTINCT d.acc, ts.term_synonym
FROM
    term AS t
    INNER JOIN graph_path AS gp ON t.id=gp.term1_id
    INNER JOIN term AS d ON d.id=gp.term2_id
    INNER JOIN term_synonym AS ts ON d.id=ts.term_id
    INNER JOIN term AS e ON ts.synonym_type_id=e.id
WHERE t.name='%%TERM_NAME%%'
    AND distance <> 0
    AND e.name='EXACT'
    AND d.is_obsolete='0';
