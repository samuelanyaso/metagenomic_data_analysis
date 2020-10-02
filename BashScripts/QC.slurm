#!/bin/sh
#SBATCH --job-name=QC					# Job name
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


##################################################################################
# Download Trimmomatic and adapter sequence files
curl -LO http://www.usadellab.org/cms/uploads/supplementary/Trimmomatic/Trimmomatic-0.36.zip
unzip Trimmomatic-0.36.zip
##################################################################################


# ############################################################
# Perform quality control using kneadData
# ############################################################

# Loads the necessary programs
module load kneaddata/0.7.4
module load bowtie2/2.3.5.1
module load trimmomatic/0.32

echo "Runs the function to download the database " 

cd /where/to/save/the/database

# Downloads the Homo_Sapiens database
kneaddata_database --download human_genome bowtie2 ./

echo "Running kneaddata script on several CPU cores" 

cd /path/to/pair_end/fastq/files
mkdir kneaddata_output

# Begins the loop for processing with knead_data
for f in $(ls *.fastq | sed -e 's/_1.fastq//' -e 's/_2.fastq//' | sort -u)
do

echo "Preprocessing ${f}"

# If files are gzipped, you unzip the files and replace the original gzipped file
# gzip -d -f ${f}_1.fastq.gz > ${f}_1.fastq
# gzip -d -f ${f}_2.fastq.gz > ${f}_2.fastq

# removal of contaminated reads and trimming
kneaddata -i ${f}_1.fastq -i ${f}_2.fastq -o kneaddata_output \
-db /path/to/database --trimmomatic /path/to/Trimmomatic-0.36 \
-t 30 --trimmomatic-options "ILLUMINACLIP:/path/to/adapter/sequence/file:2:30:10:8:keepBothReads SLIDINGWINDOW:4:30 MINLEN:60" \
--bowtie2-options "--very-sensitive --dovetail"  

echo "Finished preprocessing ${f}"
done
# End of the loop for the knead_data processing

echo "KneadData QCing complete"

# Process all of our KneadData log files and creates a nice 
# summary table with the counts of reads after each step
# Takes the kneaddata_output directory as input, output is .txt file
kneaddata_read_count_table --input kneaddata_output --output kneaddata_read_counts.txt


date


