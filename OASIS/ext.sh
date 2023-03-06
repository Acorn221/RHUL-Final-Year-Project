# Extracting the data from the tar files
for i in {1..12}
do
		filename="oasis_cross-sectional_disc${i}.tar.gz"
		tar -xvf "$filename"
done

# Copying the processed scans to a new folder
mkdir processed_scans
find . -name "*t88_masked_gfc_tra_90.gif" -exec cp {} processed_scans \;