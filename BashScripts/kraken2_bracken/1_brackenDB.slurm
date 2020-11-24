#!/bin/sh
#SBATCH --job-name=brackenDB			# Job name
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

# loads kraken2 and bracken
module load kraken/2.0.8b bracken/2.5

echo "Running bracken script on several CPU cores" 

# Generates the bracken database file
bracken-build -d /path/to/kraken2/database -t 30 -k 35 -l 150 \
	-x /path/to/kraken2/installation/directory

date
