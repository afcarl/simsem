SELECT DISTINCT d.acc, d.name
FROM
    term AS t
    INNER JOIN graph_path AS gp ON t.id=gp.term1_id
    INNER JOIN term AS d ON d.id=gp.term2_id
WHERE t.name='%%TERM_NAME%%'
    AND distance <> 0
    AND d.is_obsolete='0';
