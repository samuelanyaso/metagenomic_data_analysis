#!/bin/sh
#SBATCH --job-name=kaijuDB				# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb	            # Memory limit
#SBATCH --time=10:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out		   	# Output and error log  

pwd; hostname; date 

# loads kaiju
module load kaiju/1.7.2

echo "Creating reference database and index"

# choose <DB> (e.g. nr) from the list of database options at https://github.com/bioinformatics-centre/kaiju

kaiju-makedb -s nr

date

