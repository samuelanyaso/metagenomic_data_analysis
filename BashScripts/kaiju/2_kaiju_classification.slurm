#!/bin/sh
#SBATCH --job-name=kaiju_classn			# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb	            # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out		   	# Output and error log  

pwd; hostname; date 

# loads Kaiju
module load kaiju/1.7.2

mkdir TaxoClassn

cd /path/to/fastq/files

for f in $(ls *.fastq.gz | sed -e 's/_1.fastq.gz//' -e 's/_2.fastq.gz//' | sort -u)
do

kaiju -z 25 -t /path/to/kaijus/database/Directory/nodes.dmp -f /path/to/kaijus/database/Directory/kaiju_db_nr.fmi \
 -i ${f}_1.fastq.gz -j ${f}_2.fastq.gz \
-o ${f}.out -a mem

mv ${f}.out TaxoClassn

done

date
