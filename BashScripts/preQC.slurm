#!/bin/sh
#SBATCH --job-name=preQC				# Job name
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

# ############################################################
# Assess quality using fastQC and multiQC
# ############################################################

cd /path/to/pair_end/fastq/files
mkdir fastqc_output

# Loads the fastQC program
module load fastqc/0.11.7

# Peforms quality checks on all .fastqc files using FastQC
fastqc -t 30 *.fastq -o fastqc_output/

# Aggregate the results of fastqc quality control checks using MultiQC
cd /path/to/fastqc_output

# Loads the multiQC program
module load multiqc/1.7

# Runs the multiqc output
multiqc *_fastqc.zip

echo "pre-QC analysis complete"

date












# Begin Quality Control 

module load kneaddata/0.7.4
module load bowtie2/2.3.5.1
module load trimmomatic/0.32

echo "Running kneaddata script on several CPU cores" 

cd /ufrc/datta/sanyasosamuel/CAMDA/knead_data

# Begins the loop for processing with knead_data
for f in $(ls *.fastq | sed -e 's/_1.fastq//' -e 's/_2.fastq//' | sort -u)
do

echo "Preprocessing ${f}"

# unzips the files and replaces the original gzipped file
# gzip -d -f ${f}_1.fastq.gz > ${f}_1.fastq
# gzip -d -f ${f}_2.fastq.gz > ${f}_2.fastq

# removal of contaminated reads and trimming
kneaddata -i ${f}_1.fastq -i ${f}_2.fastq -o kneaddata_out \
-db /ufrc/datta/sanyasosamuel/CAMDA/Homo_Sapiens --trimmomatic /ufrc/datta/sanyasosamuel/CAMDA/Trimmomatic-0.36 \
-t 30 --trimmomatic-options "ILLUMINACLIP:NexteraPE-PE.fa:2:30:10:8:keepBothReads SLIDINGWINDOW:4:30 MINLEN:60" \
--bowtie2-options "--very-sensitive --dovetail"  

cd /ufrc/datta/sanyasosamuel/CAMDA/knead_data/kneaddata_out

# removes redundant files
rm *trimmed*
rm *unmatched*
rm *contam*

cd /ufrc/datta/sanyasosamuel/CAMDA/knead_data

# removes the fastq files after quality filtering
#rm ${f}_1.fastq
#rm ${f}_2.fastq
mv ${f}_1.fastq /ufrc/datta/sanyasosamuel/CAMDA/unknownSamples
mv ${f}_2.fastq /ufrc/datta/sanyasosamuel/CAMDA/unknownSamples 

echo "Finished preprocessing ${f}"

done
# End of the loop for the knead_data processing

echo "Knead-data preprocessing completed successfully and redundant files deleted"

# Process all of our KneadData log files and create a nice summary table with the counts of reads after each step
# Takes the kneaddata_out dir as input
kneaddata_read_count_table --input kneaddata_out --output CAMDA20_UNK_kneaddata_read_counts.txt

mv CAMDA20_UNK_kneaddata_read_counts.txt /ufrc/datta/sanyasosamuel/CAMDA/unknownSamples


# We begin post QC checks
cd /ufrc/datta/sanyasosamuel/CAMDA/knead_data/kneaddata_out

# removes log files from knead_data processing
rm *.log

# Loads the fastqc module
module load fastqc/0.11.7

# Peforms quality control checks on the samples using FastQC
fastqc -t 30 *paired* -o fastqc_out/

# Loads the multiqc module
module load multiqc/1.7

cd /ufrc/datta/sanyasosamuel/CAMDA/knead_data/kneaddata_out/fastqc_out

# Aggregate the results of fastqc quality control checks using MultiQC
multiqc *_fastqc.zip

# clean-up
rm *_fastqc.zip
rm *_fastqc.html
rm -r multiqc_data

# Renames the multiqc_report file
mv multiqc_report.html CAMDA20_UNK_postQC_multiqc_report.html

mv CAMDA20_UNK_postQC_multiqc_report.html /ufrc/datta/sanyasosamuel/CAMDA/unknownSamples

date


