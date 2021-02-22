#
import yaml, os, sys
import numpy as np

# import dummy_dataset from catalogue
catalogue = yaml.load(open('../zapata/catalogue.yml'), Loader=yaml.SafeLoader)

dataset = catalogue['dummy_dataset']

# create fake set of data
trgdir = dataset['path']
     

vars3 = dataset['components']['atm']['data_stream']['monthly']['3D'].keys()
vars2 = dataset['components']['atm']['data_stream']['monthly']['2D'].keys()

grid = dataset['components']['atm']['grid']
levels = dataset['levels'] 
years = dataset['year_bounds']

for vv in list(vars3) + list(vars2):
    for ll in levels:
        lev = str(ll)
        if vv in vars2:
           lev = 'SURF'
        # output path
        outdir = trgdir + '/' + vv + lev
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for yy in range(years[0],years[1]+1):
            for mm in range(1,13):
                # output filename
                mon = str(mm)
                outfile = outdir + '/' + '_'.join([vv, lev, str(yy), mon, 'MM.npy'])
                # create dummy data for variable using grid dimensions
                out = np.ones((grid['y'],grid['x']))
                if lev == 'SURF':
                    out = out * 1000. * (yy - years[0] + 1) + mm 
                else:
                    out = out * int(lev) * (yy - years[0] + 1) + mm 
                # save to numpy data format
                np.save(outfile, out, allow_pickle=False)
                del(outfile, out)

        # skip level loop for 2D vars
        if vv in vars2:
            break
       


                


