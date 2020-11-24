#!/bin/bash
#SBATCH --job-name=AbundanceEstimation  # Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=30gb             	# Memory limit
#SBATCH --time=02:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out   		# Output and error log 

pwd; hostname; date 

# loads bracken
module load bracken/2.5

cd /path/to/kraken2/report/files

for f in $(ls *.kreport | sed -e 's/.kreport//' | sort -u)
do
# Estimate abundance with Bracken
bracken -d /path/to/kraken2/database -i ${f}.kreport -o ${f}.bracken -r 150 \
	-l S -t 10
done

echo "Estimation of species abundance with bracken complete"

date
