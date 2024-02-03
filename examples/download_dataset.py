import nnetsauce as ns 

# the controversial Boston dataset 
df1 = ns.Downloader().download(dataset="Boston")

print(f"===== df1: \n {df1} \n")
print(f"===== df1.dtypes: \n {df1.dtypes}")

print("\n====================================================== \n")

# Insurance dataset 
df2 = ns.Downloader().download(dataset="Insurance")
print(f"===== df2: \n {df2} \n")
print(f"===== df2.dtypes: \n {df2.dtypes}")

print("\n====================================================== \n")

# Affairs dataset
df3 = ns.Downloader().download(pkgname="AER", dataset="Affairs", source="https://zeileis.r-universe.dev/")
print(f"===== df3: \n {df3} \n")
print(f"===== df3.dtypes: \n {df3.dtypes}")
