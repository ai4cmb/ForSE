import numpy as np

def bin_array(array, bins=100):
    len_data = len(array)
    x = np.arange(len_data)+1
    num_bin = len_data//bins
    data_binned = []
    x_binned = []
    for i in range(num_bin):
        data_binned.append(np.mean(array[i*bins:(i+1)*bins]))
        x_binned.append(np.mean(x[i*bins:(i+1)*bins]))
    data_binned = np.array(data_binned)
    x_binned = np.array(x_binned)
    return x_binned, data_binned


def estimate_marchingsquare(data , threshold ):
    width = data.shape[0]
    height= data.shape[1]
    f,u,chi=0 ,0,0
    for i in range(width-1 ):
        for j in range(height-1):
            pattern=0
            if (data[i,j]     > threshold) : pattern += 1;
            if (data[i+1,j]   > threshold) : pattern += 2;
            if (data[i+1,j+1] > threshold) : pattern += 4;
            if (data[i,j+1 ]  > threshold) : pattern += 8;
            if pattern ==0 :
                break
            elif pattern==1:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j])
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi + 0.25;
                break;
            elif pattern==2:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1, (j+1)]);
                f = f + 0.5 * (1 - a1) * (a2);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi + 0.25;
                break;
            elif pattern==3:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,(j+1)]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + a2 + 0.5 * (a4 - a2);
                u = u + np.sqrt(1 + (a4 - a2) * (a4 - a2));
                break;
            elif pattern==4:
                a2 = (data[ i+1,j] - threshold) / (data[i+1,j ] - data[ i+1,j+1]);
                a3 = (data[ i,j+1 ] -  threshold) / (data[i,j+1] - data[ i+1,j+1]);
                f = f + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.25;
                break;
            elif pattern==5:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.5;
                break;
            elif pattern==6:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + (1 - a3) + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                break;
            elif pattern==7:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi - 0.25;
                break;

            elif pattern==8:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.25;
                break;
            elif pattern==9:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + a1 + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                break;
            elif pattern==10:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt(a1 * a1 + a4 * a4) + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.5;
                break;
            elif pattern==11:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + 1 - 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi - 0.25;
                break;
            elif pattern==12:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + (1 - a2) + 0.5 * (a2 - a4);
                u = u + np.sqrt(1 + (a2 - a4) * (a2 - a4));
                break;
            elif pattern==13:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                f = f + 1 - .5 * (1 - a1) * a2;
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi - 0.25;
                break;
            elif pattern==14:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi - 0.25;
                break;
            elif pattern == 15:
                f +=1 ;
                break;


    return f,u, chi

def get_functionals(im , nevals= 32):
    vmin =im.min() ; vmax=im.max()
    rhos =  np.linspace( vmin,vmax, nevals)
    f= np.zeros_like(rhos)
    u= np.zeros_like(rhos)
    chi= np.zeros_like(rhos)
    for k, rho in np.ndenumerate( rhos) :
        f[k], u[k],chi[k]=  estimate_marchingsquare(im, rho )
    return rhos, f,u,chi

def compute_intersection(x, cont1, cont2, npt=10000):
    ymin1 = cont1[0]
    ymax1 = cont1[1]
    ymin2 = cont2[0]
    ymax2 = cont2[1]
    yMAX = np.max([ymax1, ymax2])+0.1*np.max([ymax1, ymax2])
    yMIN = np.min([ymin1, ymin2])-0.1*np.min([ymin1, ymin2])
    area1 = 0
    area2 = 0
    areaint = 0
    ind_xi = np.random.randint(0, len(x), npt)
    yi = np.random.uniform(yMIN, yMAX, npt)
    for i in range(npt):
        if ymin1[ind_xi[i]]<=yi[i]<=ymax1[ind_xi[i]]:
            area1 += 1
            if ymin2[ind_xi[i]]<=yi[i]<=ymax2[ind_xi[i]]:
                areaint += 1
        elif ymin2[ind_xi[i]]<=yi[i]<=ymax2[ind_xi[i]]:
            area2 += 1
    return areaint/(area1+area2)
