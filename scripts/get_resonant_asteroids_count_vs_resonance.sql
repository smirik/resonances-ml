select count(*) as ast_count, m1, m2, m
from librations
where 
pure=1 AND planet1='JUPITER' AND planet2='SATURN'
group by m1, m2, m HAVING ast_count >= 50 order by ast_count;
