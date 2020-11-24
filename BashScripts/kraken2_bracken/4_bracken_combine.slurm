#!/bin/bash
#SBATCH --job-name=bracken_combine    	# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 1		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb             	# Memory limit
#SBATCH --time=02:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out   		# Output and error log  

pwd; hostname; date 

# loads bracken 
module load bracken/2.5

echo "Combining bracken output files " 

cd /path/to/bracken/output/files

combine_bracken_outputs.py --files *.bracken -o output_file

date
