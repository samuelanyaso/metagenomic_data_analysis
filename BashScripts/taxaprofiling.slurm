#!/bin/sh
#SBATCH --job-name=taxaprofiling		# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=500mb             # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=serial_test_%j.out   	# Output and error log 

pwd; hostname; date 

# loads MetaPhlAn2 
module load metaphlan2/2.96.1

echo "Running metaphlan2 script on several CPU cores" 

cd /path/to/fastq/files

# Perform taxonomic profiling of multiple .fastq files using MetaPhlAn2
for f in $(ls *.fastq | sed -e 's/_1.fastq//' -e 's/_2.fastq//' | sort -u)
do

metaphlan2.py --bowtie2db /path/to/metaphlan_databases ${f}_1.fastq,${f}_2.fastq --bowtie2out ${f}.bt2out --nproc 30 --input_type fastq 

metaphlan2.py --bowtie2db /path/to/metaphlan_database ${f}.bt2out --nproc 30 --input_type bowtie2out > ${f}_profile.txt

done

# Merge taxonomic profiles for each sample into a single .txt file
merge_metaphlan_tables.py *_profile.txt > merged_abundance_table.txt

date 