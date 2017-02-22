#!/bin/bash
# Gets numbers of pure resonant asteroid numbers and place them to files grouping by resonance.
mysql $1 -u $2 -p$3 < get_resonant_asteroids.sql | sed 's/\t/;/g' > /tmp/asts.txt
tail -n +2 /tmp/asts.txt > /tmp/asts.tmp.txt
mv /tmp/asts.tmp.txt /tmp/asts.txt
for i in `cat /tmp/asts.txt`; do echo $i | cut -f1 -d';' >> "JUPITER-SATURN_""`echo $i | cut -f5 -d';'`"; done
