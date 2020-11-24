#!/bin/sh
#SBATCH --job-name=metaphlan2			# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=2gb             # Memory limit
#SBATCH --time=02:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out   	# Output and error log 

pwd; hostname; date 

# loads MetaPhlAn2 
module load metaphlan2/2.96.1

cd /path/to/fastq/files

# Perform taxonomic profiling of multiple .fastq files using MetaPhlAn2
for f in $(ls *.fastq | sed -e 's/_1.fastq//' -e 's/_2.fastq//' | sort -u)
do
metaphlan2.py --bowtie2db /path/to/metaphlan_databases ${f}_1.fastq,${f}_2.fastq --bowtie2out ${f}.bt2out --nproc 30 --input_type fastq 

metaphlan2.py --bowtie2db /path/to/metaphlan_database ${f}.bt2out --nproc 30 --input_type bowtie2out > ${f}_profile.txt
done

date 