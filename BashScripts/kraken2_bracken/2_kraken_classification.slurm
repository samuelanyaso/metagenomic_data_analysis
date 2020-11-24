#!/bin/bash
#SBATCH --job-name=krakenClassification # Job name
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

# loads kraken2 
module load kraken/2.0.8b

cd /path/to/preprocessed/samples

for f in $(ls *.fastq.gz | sed -e 's/_1.fastq.gz//' -e 's/_2.fastq.gz//' | sort -u)
do
# Generate kraken2 report files
kraken2 --db  /path/to/kraken2/database --threads 30 --report ${f}.kreport \
	--fastq-input --gzip-compressed --paired ${f}_1.fastq.gz ${f}_2.fastq.gz \
	> ${f}.kraken
done

echo "Classification with kraken complete"

date
