import nnetsauce as ns 


# Insurance dataset 
df2 = ns.Downloader().download(dataset="Insurance")
print(f"===== df2: \n {df2} \n")
print(f"===== df2.dtypes: \n {df2.dtypes}")

print("\n====================================================== \n")
