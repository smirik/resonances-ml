-- Gets numbers of pure resonant asteroid numbers.
SELECT asteroid, m1, m2, m, concat(m1, '_', m2, '_', m) AS iview
FROM librations
WHERE 
pure=1 AND asteroid regexp '^[0-9]*$' AND planet1='JUPITER' AND planet2='SATURN' -- get only pure numbered JS
AND concat(m1, '_', m2, '_', m) in (
    SELECT concat(m1, '_', m2, '_', m) as `view`
    FROM librations
    WHERE 
    pure=1 AND asteroid regexp '^[0-9]*$' AND planet1='JUPITER' AND planet2='SATURN' -- get only pure numbered JS
    GROUP BY m1, m2, m HAVING count(*) >= 50 order by count(*)
)
ORDER BY iview, cast(asteroid AS UNSIGNED);
