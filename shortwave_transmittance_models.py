import numpy as np
import pandas as pd
from pyTSEB import meteo_utils as met
from pyTSEB import TSEB
from pyTSEB import net_radiation as rad
from py3seb.py3seb import calc_shadow_fraction
from py3seb.py3seb import ThreeSEB_PT, raupach_94, calc_Sn_Campbell


grados_2_rad = lambda x: np.cos(x * np.pi / 180)

folder = rf'C:\Users\mqalborn\Desktop\GRAPEX\RIP\outcomes/biophysical_traits.csv'
data_bt = pd.read_csv(folder)
spectraGrd = {'rsoilv': 0.07, 'rsoiln': 0.28}
spectraVeg_tree = {'rho_leaf_vis': 0.096, 'tau_leaf_vis': 0.014, 'rho_leaf_nir': 0.55, 'tau_leaf_nir': 0.28}
spectraVeg_grass = {'rho_leaf_vis': 0.05, 'tau_leaf_vis': 0.09, 'rho_leaf_nir': 0.24, 'tau_leaf_nir': 0.43}


# Setting overstory vegetation traits
lai_ov = np.array(data_bt.LAI_ov)
fc_ov = np.array(data_bt.Fov)
x_LAD_ov = np.full_like(lai_ov, 1)
hc_ov = np.array(- data_bt.Hc)
wc_ov = np.array(data_bt.Wov)
wc_hc_ratio = wc_ov / hc_ov

hb_ov = np.full_like(lai_ov, 0.5)

# Setting understory vegetation traits
lai_un = np.array(data_bt.LAI_un)
lai_un[np.isnan(lai_un)] = 0

x_LAD_un = np.full_like(lai_ov, 1)
x_LAD_un[lai_un==0] = 0

fc_un = np.array(data_bt.Fun)
fc_un[lai_un==0] = 0

wc_un = np.full_like(lai_ov, 1)
wc_un[lai_un==0] = 0

# Weather data
S_dn = np.array(data_bt.SW_IN)
atmPress = np.full_like(lai_ov, 1013.25)
# Location info
lat = np.full_like(lai_ov, 36.84)
lon = np.full_like(lai_ov, -120.1758)
stdlon = np.full_like(lai_ov, -120)
row_direction = np.full_like(lai_ov, 88)

# flight info
doy = np.array(pd.to_datetime(data_bt.date).dt.day_of_year)
ftime = np.array(data_bt.hour)

# Sun Angles and Daylength
sza, saa = met.calc_sun_angles(lat=lat, lon=lon, stdlon=stdlon, doy=doy, ftime=ftime)
psi = row_direction - saa

# Estimating Direct and Diffuse Short-wave Irradiance
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(S_dn=S_dn, sza=sza, press=atmPress)
skyl = fvis * difvis + fnir * difnir
Sdn_dir = (1. - skyl) * S_dn
Sdn_dif = skyl * S_dn
print(Sdn_dif)

rho_leaf_vis_ov = np.full_like(lai_ov, spectraVeg_tree['rho_leaf_vis'])
rho_leaf_nir_ov = np.full_like(lai_ov, spectraVeg_tree['rho_leaf_nir'])

tau_leaf_vis_ov = np.full_like(lai_ov, spectraVeg_tree['tau_leaf_vis'])
tau_leaf_nir_ov = np.full_like(lai_ov, spectraVeg_tree['tau_leaf_vis'])

rho_leaf_vis_un = np.full_like(lai_ov, spectraVeg_grass['rho_leaf_vis'])
rho_leaf_nir_un = np.full_like(lai_ov, spectraVeg_grass['rho_leaf_nir'])

tau_leaf_vis_un = np.full_like(lai_ov, spectraVeg_grass['tau_leaf_vis'])
tau_leaf_nir_un = np.full_like(lai_ov, spectraVeg_grass['tau_leaf_vis'])

rsoilv = np.full_like(lai_ov, spectraGrd['rsoilv'])
rsoiln = np.full_like(lai_ov, spectraGrd['rsoiln'])

rho_leaf_ov = np.array((rho_leaf_vis_ov, rho_leaf_nir_ov))
tau_leaf_ov = np.array((tau_leaf_vis_ov, tau_leaf_nir_ov))

rho_leaf_un = np.array((rho_leaf_vis_un, rho_leaf_nir_un))
tau_leaf_un = np.array((tau_leaf_vis_un, tau_leaf_nir_un))

rho_soil = np.array((rsoilv, rsoiln))

# Campbell and Norman Radiation Transfer Model
## Overstory Clumping Index

### Rectangular Clumping Index
omega0_ov = TSEB.CI.calc_omega0_Kustas(lai_ov, fc_ov, x_LAD=1, isLAIeff=True)
omegaH_ov = TSEB.CI.calc_omega_Kustas(omega0_ov, sza, w_C=wc_hc_ratio)
lai_ov_eff_H = lai_ov * omegaH_ov

omegaR_ov = TSEB.CI.calc_omega_rows(lai=lai_ov,
                                    f_c0=fc_ov,
                                    theta=sza,
                                    psi=psi,
                                    w_c=wc_hc_ratio,
                                    x_lad=1,
                                    is_lai_eff=False)
lai_ov_eff_R = lai_ov * omegaR_ov

### Inital estimate with Rho_soil to get Tree transmittance
_, _, taubt_ov0, taudt_ov0 = rad.calc_spectra_Cambpell(
    lai=lai_ov,
    sza=sza,
    rho_leaf=rho_leaf_ov,
    tau_leaf=tau_leaf_ov,
    rho_soil=rho_soil,
    x_lad=x_LAD_ov,
    lai_eff=lai_ov_eff_R)

### Understory reflectance and transmittance
#### Transmission of Radiation by Sparse Canopies - Soil Reflectance Effects
#### No Clumping Index.
albb_un, albd_un, taubt_un, taudt_un = rad.calc_spectra_Cambpell(
    lai=lai_un,
    sza=sza,
    rho_leaf=rho_leaf_un,
    tau_leaf=tau_leaf_un,
    rho_soil=rho_soil,
    x_lad=x_LAD_un,
    lai_eff=None)

#### Clumping Index.
Omega0_un = TSEB.CI.calc_omega0_Kustas(lai_un, fc_un, x_LAD=x_LAD_un, isLAIeff=True)
Omega_un = TSEB.CI.calc_omega_Kustas(Omega0_un, sza, w_C=1)
lai_un_eff = lai_un * Omega_un

albb_un_H, albd_un_H, taubt_un_H, taudt_un_H = rad.calc_spectra_Cambpell(
    lai=lai_un,
    sza=sza,
    rho_leaf=rho_leaf_un,
    tau_leaf=tau_leaf_un,
    rho_soil=rho_soil,
    x_lad=x_LAD_un,
    lai_eff=lai_un_eff)

### get percent of shaded area on substrate
f_shaded = calc_shadow_fraction(sza=sza, hc=hc_ov, hb=hb_ov, wc=wc_hc_ratio, f_C=fc_ov, tau=np.mean(taubt_ov0))
### the direct substrate albedo dominates over the overstory gaps and there are no shadows
rho_soil_un = f_shaded * albd_un + (1 - f_shaded) * albb_un

### Overstory reflectance and transmittance
#### Transmission of Radiation by Sparse Canopies - Soil Reflectance Effects

#### Taking Into Account Homogeneous Clumping Index
albb_ov_H, albd_ov_H, taubt_ov_H, taudt_ov_H = rad.calc_spectra_Cambpell(
    lai=lai_ov,
    sza=sza,
    rho_leaf=rho_leaf_un,
    tau_leaf=tau_leaf_un,
    rho_soil=rho_soil_un,
    x_lad=x_LAD_ov,
    lai_eff=lai_ov_eff_H
)

#### Taking Into Account Rectangular Clumping Index
albb_ov_R, albd_ov_R, taubt_ov_R, taudt_ov_R = rad.calc_spectra_Cambpell(
    lai=lai_ov,
    sza=sza,
    rho_leaf=rho_leaf_un,
    tau_leaf=tau_leaf_un,
    rho_soil=rho_soil_un,
    x_lad=x_LAD_ov,
    lai_eff=lai_ov_eff_R
)

### Overstory net shortwave radiation
T_ov_H = ((taubt_ov_H[0] * Sdn_dir * fvis) +
          (taubt_ov_H[1] * Sdn_dir * fnir) +
          (taudt_ov_H[0] * Sdn_dif * fvis) +
          (taudt_ov_H[1] * Sdn_dif * fnir))

T_ov_R = ((taubt_ov_R[0] * Sdn_dir * fvis) +
          (taubt_ov_R[1] * Sdn_dir * fnir) +
          (taudt_ov_R[0] * Sdn_dif * fvis) +
          (taudt_ov_R[1] * Sdn_dif * fnir))

Sn_ov_H = ((1.0 - taubt_ov_H[0]) * (1.0 - albb_ov_H[0]) * Sdn_dir * fvis
        + (1.0 - taubt_ov_H[1]) * (1.0 - albb_ov_H[1]) * Sdn_dir * fnir
        + (1.0 - taudt_ov_H[0]) * (1.0 - albd_ov_H[0]) * Sdn_dif * fvis
        + (1.0 - taudt_ov_H[1]) * (1.0 - albd_ov_H[1]) * Sdn_dif * fnir)

Sn_ov_R = ((1.0 - taubt_ov_R[0]) * (1.0 - albb_ov_R[0]) * Sdn_dir * fvis
        + (1.0 - taubt_ov_R[1]) * (1.0 - albb_ov_R[1]) * Sdn_dir * fnir
        + (1.0 - taudt_ov_R[0]) * (1.0 - albd_ov_R[0]) * Sdn_dif * fvis
        + (1.0 - taudt_ov_R[1]) * (1.0 - albd_ov_R[1]) * Sdn_dif * fnir)

Sdn_un_dir_vis = taubt_ov_R[0] * Sdn_dir * fvis
Sdn_un_dif_vis = taudt_ov_R[0] * Sdn_dif * fvis
Sdn_un_dir_nir = taubt_ov_R[1] * Sdn_dir * fnir
Sdn_un_dif_nir = taudt_ov_R[1] * Sdn_dif * fnir

### Understory net shortwave radiation
#### No clumping index
Sn_un = ((1.0 - taubt_un[0]) * (1.0 - albb_un[0]) * Sdn_un_dir_vis
            + (1.0 - taubt_un[1]) * (1.0 - albb_un[1]) * Sdn_un_dir_nir
            + (1.0 - taudt_un[0]) * (1.0 - albd_un[0]) * Sdn_un_dif_vis
            + (1.0 - taudt_un[1]) * (1.0 - albd_un[1]) * Sdn_un_dif_nir)

#### Clumping Index
Sn_un_H = ((1.0 - taubt_un_H[0]) * (1.0 - albb_un_H[0]) * Sdn_un_dir_vis
            + (1.0 - taubt_un_H[1]) * (1.0 - albb_un_H[1]) * Sdn_un_dir_nir
            + (1.0 - taudt_un_H[0]) * (1.0 - albd_un_H[0]) * Sdn_un_dif_vis
            + (1.0 - taudt_un_H[1]) * (1.0 - albd_un_H[1]) * Sdn_un_dif_nir)

Sn_un[lai_un<=0] = np.nan
Sn_un_H[lai_un<=0] = np.nan

### Soil net shortwave radiation
Sn_s = (taubt_un[0] * (1.0 - rsoilv) * Sdn_un_dir_vis
        + taubt_un[1] * (1.0 - rsoiln) * Sdn_un_dir_nir
        + taudt_un[0] * (1.0 - rsoilv) * Sdn_un_dif_vis
        + taudt_un[1] * (1.0 - rsoiln) * Sdn_un_dif_nir)
#### Clumping Index
Sn_s_H = (taubt_un_H[0] * (1.0 - rsoilv) * Sdn_un_dir_vis
        + taubt_un_H[1] * (1.0 - rsoiln) * Sdn_un_dir_nir
        + taudt_un_H[0] * (1.0 - rsoilv) * Sdn_un_dif_vis
        + taudt_un_H[1] * (1.0 - rsoiln) * Sdn_un_dif_nir)

data_bt.loc[:, 'Sn_ov_H'] = Sn_ov_H
data_bt.loc[:, 'Sn_ov_R'] = Sn_ov_R

data_bt.loc[:, 'T_ov_H'] = T_ov_H
data_bt.loc[:, 'T_ov_R'] = T_ov_R

data_bt.loc[:, 'Sn_un'] = Sn_un
data_bt.loc[:, 'Sn_un_H'] = Sn_un_H
data_bt.loc[:, 'Sn_s'] = Sn_s
data_bt.loc[:, 'Sn_s_H'] = Sn_s_H

data_bt_sn = data_bt[['block', 'transect', 'tree', 'hour',
                      'Sn_ov_H', 'Sn_ov_R', 'T_ov_H', 'T_ov_R',
                      'Sn_un', 'Sn_un_H', 'Sn_s', 'Sn_s_H']]

folder = rf'C:\Users\mqalborn\Desktop\GRAPEX\RIP\outcomes/shortwave_transmittance.csv'
data_bt_sn.to_csv(folder)

"""
Sn_ov_R, Sn_S = rad.calc_Sn_Campbell(lai=lai_ov,
                 sza=sza,
                 S_dn_dir=Sdn_dir,
                 S_dn_dif=Sdn_dif,
                 fvis=fvis,
                 fnir=fnir,
                 rho_leaf_vis=rho_leaf_vis_ov,
                 tau_leaf_vis=tau_leaf_vis_ov,
                 rho_leaf_nir=rho_leaf_nir_ov,
                 tau_leaf_nir=tau_leaf_nir_ov,
                 rsoilv=rsoilv,
                 rsoiln=rsoiln,
                 x_LAD=1,
                 LAI_eff=lai_ov_eff_R)

Sn_ov_H, Sn_S = rad.calc_Sn_Campbell(lai=lai_ov,
                 sza=sza,
                 S_dn_dir=Sdn_dir,
                 S_dn_dif=Sdn_dif,
                 fvis=fvis,
                 fnir=fnir,
                 rho_leaf_vis=rho_leaf_vis_ov,
                 tau_leaf_vis=tau_leaf_vis_ov,
                 rho_leaf_nir=rho_leaf_nir_ov,
                 tau_leaf_nir=tau_leaf_nir_ov,
                 rsoilv=rsoilv,
                 rsoiln=rsoiln,
                 x_LAD=1,
                 LAI_eff=lai_ov_eff_H)
"""




#
#
# ### Inital estimate with Rho_soil to get Tree transmittance
# _, _, taubt, taudt = rad.calc_spectra_Cambpell(lai,
#                                                sza,
#                                                rho_leaf,
#                                                tau_leaf,
#                                                rho_soil,
#                                                x_lad=x_LAD,
#                                                lai_eff=LAI_eff)
#
#
#
# albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
#                                                      sza,
#                                                      rho_leaf,
#                                                      tau_leaf,
#                                                      rho_sub,
#                                                      x_lad=x_LAD,
#                                                      lai_eff=LAI_eff)
#
# S_sub_dir_vis = taubt[0] * S_dn_dir * fvis
# S_sub_dif_vis = taudt[0] * S_dn_dif * fvis
# S_sub_dir_nir = taubt[1] * S_dn_dir * fnir
# S_sub_dif_nir = taudt[1] * S_dn_dif * fnir
#
# Sn_sub_C = ((1.0 - taubt_sub[0]) * (1.0 - albb_sub[0]) * S_sub_dir_vis
#             + (1.0 - taubt_sub[1]) * (1.0 - albb_sub[1]) * S_sub_dir_nir
#             + (1.0 - taudt_sub[0]) * (1.0 - albd_sub[0]) * S_sub_dif_vis
#             + (1.0 - taudt_sub[1]) * (1.0 - albd_sub[1]) * S_sub_dif_nir)
#
# Sn_S = (taubt_sub[0] * (1.0 - rsoilv) * S_sub_dir_vis
#         + taubt_sub[1] * (1.0 - rsoiln) * S_sub_dir_nir
#         + taudt_sub[0] * (1.0 - rsoilv) * S_sub_dif_vis
#         + taudt_sub[1] * (1.0 - rsoiln) * S_sub_dif_nir)
#
# print(albb_ov)
