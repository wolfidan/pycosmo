import numpy as np
import pycosmo as pc

from scipy.interpolate import interp1d
from scipy.integrate import odeint

RAD_TO_DEG = 180.0/np.pi
DEG_TO_RAD = 1./RAD_TO_DEG


class _Beam():
    def __init__(self, dic_values, mask, lats_profile,lons_profile, dist_ground_profile,heights_profile, GH_pt=[],GH_weight=1):
        self.mask=mask
        self.GH_pt=GH_pt
        self.GH_weight=GH_weight
        self.lats_profile=lats_profile
        self.lons_profile=lons_profile
        self.dist_profile=dist_ground_profile
        self.heights_profile=heights_profile
        self.values=dic_values
    
def _sum_arr(x,y, cst = 0):
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))          
            pad_1.append((0,0))
            
        
    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)
    
    z = np.sum([x,y],axis=0)
    
    return z
    
def _nansum_arr(x,y, cst = 0):

    x = np.array(x)
    y = np.array(y)
    
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))          
            pad_1.append((0,0))
        
    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)
    
    z = np.nansum([x,y],axis=0)
    return z    
    
def _piecewise_linear(x,y):
    interpolator=interp1d(x,y)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if np.isscalar(xs):
            xs=[xs]
        return np.array([pointwise(xi) for xi in xs])
        
    return ufunclike

def integrate_quad(list_GH_pts):
    n_beams = len(list_GH_pts)
    n_gates = len(list_GH_pts[0].mask)
    list_variables = list_GH_pts[0].values.keys()
 
    # Sum the mask of all beams to get overall mask
    mask = np.zeros(n_gates,) # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    frac_pow = np.zeros(n_gates,) # Fraction of power received at antenna
    for i,p in enumerate(list_GH_pts):
        mask = _sum_arr(mask,p.mask) # Get mask of every Beam
        frac_pow = _sum_arr(frac_pow, (p.mask==0).astype(float)*p.GH_weight)

    mask/=float(n_beams) #mask == 1 means that every Beam is below TOPO, 
                            #smaller than 0 that at least one Beam is above COSMO domain
    
    integrated_variables={}
    for k in list_variables:
        integrated_variables[k]=[float('nan')]
        for p in list_GH_pts:
            integrated_variables[k] = _nansum_arr(integrated_variables[k],
                                        p.values[k]*p.GH_weight)
        
        integrated_variables[k][np.logical_or(mask>1,mask<=-1)] = float('nan')
        integrated_variables[k] /= frac_pow # Normalize by valid pts
        
    # Get index of central beam
    idx_0=int(n_beams/2)

    heights_radar=list_GH_pts[idx_0].heights_profile
    distances_radar=list_GH_pts[idx_0].dist_profile
    lats=list_GH_pts[idx_0].lats_profile
    lons=list_GH_pts[idx_0].lons_profile

    integrated_beam = _Beam(integrated_variables,mask,lats, lons, distances_radar, heights_radar)
    return integrated_beam, frac_pow

    
def quad_pts_weights(beamwidth, npts_GH):
        
    # Calculate quadrature weights and points
    nh_GH = npts_GH[0]
    nv_GH = npts_GH[1]
        
    # Get GH points and weights
    sigma = beamwidth/(2*np.sqrt(2*np.log(2)))

    pts_hor, weights_hor=np.polynomial.hermite.hermgauss(nh_GH)
    pts_hor=pts_hor*sigma
   
    pts_ver, weights_ver=np.polynomial.hermite.hermgauss(nv_GH)
    pts_ver=pts_ver*sigma

    weights = np.outer(weights_hor*sigma,weights_ver*sigma)
    weights *= np.abs(np.cos(pts_ver))
    
    sum_weights = np.sum(weights.ravel())

    return [pts_hor,pts_ver], weights/sum_weights
        
def earth_radius(latitude):
    a=6378.1370*1000 # largest radius
    b=6356.7523*1000 # smallest radius
    return np.sqrt(((a**2*np.cos(latitude))**2+\
                    (b**2*np.sin(latitude))**2)/((a*np.cos(latitude))**2\
                    +(b*np.sin(latitude))**2))

def refraction_sh(range_vec, elevation_angle, coords_radar, refraction_method, N=0):
    # Method can be '4/3', 'ODE_s'
    # '4/3': Standard 4/3 refraction model (offline, very fast)
    # ODE_s: differential equation of atmospheric refraction assuming horizontal homogeneity

    if refraction_method==1:
        S,H = ref_4_3(range_vec, elevation_angle, coords_radar)
    elif refraction_method==2:
        S,H = ref_ODE_s(range_vec, elevation_angle, coords_radar, N)

    return S,H  
    
def deriv_z(z,r,n_h_spline, dn_dh_spline, RE):
    # Computes the derivatives (RHS) of the system of ODEs
    h,u=z
    n=n_h_spline(h)
    dn_dh=dn_dh_spline(h)
    return [u, -u**2*((1./n)*dn_dh+1./(RE+h))+((1./n)*dn_dh+1./(RE+h))]
    
def ref_ODE_s(range_vec, elevation_angle, coords_radar, N):
    
    # Get info about COSMO coordinate system
    proj_COSMO = N.attributes['proj_info']
    coords_rad_in_COSMO = pc.WGS_to_COSMO(coords_radar,
                              [proj_COSMO['Latitude_of_southern_pole'],
                               proj_COSMO['Longitude_of_southern_pole']])
    
    llc_COSMO=(float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    res_COSMO=N.attributes['resolution']
    
    # Get position of radar in COSMO coordinates
     # Note that for lat and lon we stay with indexes but for the vertical we have real altitude s
    pos_radar_bin=[(coords_rad_in_COSMO[0]-llc_COSMO[1])/res_COSMO[1],
                   (coords_rad_in_COSMO[1]-llc_COSMO[0])/res_COSMO[0]]
                   
    # Get refractive index profile from refractivity estimated from COSMO variables
    n_vert_profile=1+(N.data[:,int(np.round(pos_radar_bin[0])),
                             int(np.round(pos_radar_bin[0]))])*1E-6
    # Get corresponding altitudes
    h = N.attributes['z-levels'][:,int(np.round(pos_radar_bin[0])),
                                int(np.round(pos_radar_bin[0]))] 
    
    # Get earth radius at radar latitude
    RE = earth_radius(coords_radar[0])

    # Invert to get from ground to top of model domain
    h=h[::-1]
    n_vert_profile=n_vert_profile[::-1] # Refractivity
        
    # Create piecewise linear interpolation for n as a function of height
    n_h_spline = _piecewise_linear(h, n_vert_profile)
    dn_dh_spline = _piecewise_linear(h[0:-1],np.diff(n_vert_profile)/np.diff(h))
    
    z_0 = [coords_radar[2],np.deg2rad(elevation_angle)]
    # Solve second-order ODE
    Z = odeint(deriv_z,z_0,range_vec,args=(n_h_spline,dn_dh_spline,RE))
    H = Z[:,0] # Heights above ground
    E = Z[:,1] # Elevations
    S = np.zeros(H.shape) # Arc distances
    dR = range_vec[1]-range_vec[0]
    S[0] = 0
    for i in range(1,len(S)): # Solve for arc distances
        S[i] = S[i-1]+RE*np.arcsin((np.cos(E[i-1])*dR)/(RE+H[i]))

    return S.astype('float32'), H.astype('float32')
    
def ref_4_3(range_vec, elevation_angle, coords_radar):
    elevation_angle=elevation_angle*np.pi/180.
    ke=4./3.
    altitude_radar=coords_radar[2]
    latitude_radar=coords_radar[1]
    # Compute earth radius at radar latitude 
    EarthRadius = earth_radius(latitude_radar)
    # Compute height over radar of every range_bin        
    H=np.sqrt(range_vec**2 + (ke*EarthRadius)**2+2*range_vec*ke*EarthRadius*np.sin(elevation_angle))-ke*EarthRadius+altitude_radar
    # Compute arc distance of every range bin
    S=ke*EarthRadius*np.arcsin((range_vec*np.cos(elevation_angle))/(ke*EarthRadius+H))
    
    return S.astype('float32'),H.astype('float32')