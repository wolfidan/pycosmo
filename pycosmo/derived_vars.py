#!/usr/bin/env python2
import numpy as np

# COSMO CONSTANTS
COSMO_R_D = 287.05
COSMO_R_V = 451.51
COSMO_RDV = COSMO_R_D / COSMO_R_V
COSMO_O_M_RDV = 1.0 - COSMO_RDV
COSMO_RVD_M_O = COSMO_R_V / COSMO_R_D - 1.0


DERIVED_VARS=['PREC_RATE','QV_v','QR_v','QS_v','QG_v','QC_v','QI_v','QH_v',
              'QNR_v','QNS_v','QNG_v','QNC_v','QNI_v','QNH_v',
              'LWC','TWC','IWC','RHO','N','Pw','RELHUM']
              
    
def get_derived_var(file_instance, varname, options):
    derived_var=None
    try:
        if varname == 'PREC_RATE': # PRECIPITATION RATE
            vars_to_get = ['PRR_GSP_GDS10_SFC','PRR_CON_GDS10_SFC','PRS_CON_GDS10_SFC','PRS_GSP_GDS10_SFC']
            d = file_instance.get_variable(vars_to_get,**options)
            derived_var=d['PRR_GSP_GDS10_SFC']+d['PRR_CON_GDS10_SFC']+\
                        d['PRS_CON_GDS10_SFC']+d['PRS_GSP_GDS10_SFC']
            if 'PRG_GSP_GDS10_SFC' in file_instance.variables.keys(): # Check if graupel is present
                derived_var += file_instance.get_variable('PRG_GSP_GDS10_SFC',**options)
            if 'PRH_GSP_GDS10_SFC' in file_instance.variables.keys(): # Check if hail is present
                derived_var += file_instance.get_variable('PRH_GSP_GDS10_SFC',**options)
            derived_var.name='PREC_RATE'
            derived_var.attributes['long_name']='precipitation intensity'
            derived_var.attributes['units'] = 'mm/s'
            
        elif varname == 'QV_v': # Water vapour mass density
            d = file_instance.get_variable(['QV','RHO'],**options)
            derived_var = d['QV']*d['RHO']
            derived_var.name = 'QV_v'
            derived_var.attributes['units'] = 'kg/m3'
            derived_var.attributes['long_name'] = 'Water vapor mass density'
            
        elif varname == 'QR_v': # Rain water mass density
            d = file_instance.get_variable(['QR','RHO'],**options)
            derived_var = d['QR']*d['RHO']
            derived_var.name='QR_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Rain mass density'
            
        elif varname == 'QS_v': # Snow water mass density
            d = file_instance.get_variable(['QS','RHO'],**options)
            derived_var=d['QS']*d['RHO']
            derived_var.name='QS_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Snow mass density'
            
        elif varname == 'QG_v': # Graupel water mass density
            d = file_instance.get_variable(['QG','RHO'],**options)
            derived_var=d['QG']*d['RHO']
            derived_var.name='QG_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Graupel mass density'
            
        elif varname == 'QC_v': # Cloud water mass density
            d = file_instance.get_variable(['QC','RHO'],**options)
            derived_var=d['QC']*d['RHO']
            derived_var.name='QC_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Cloud mass density'
            
        elif varname == 'QI_v': # Ice cloud water mass density
            d = file_instance.get_variable(['QI','RHO'],**options)
            derived_var=d['QI']*d['RHO']
            derived_var.name='QI_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Ice crystals mass density'        
            
        elif varname == 'QH_v': # Hail water mass density
            d = file_instance.get_variable(['QH','RHO'],**options)
            derived_var=d['QH']*d['RHO']
            derived_var.name='QH_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Hail mass density'
            
        elif varname == 'QNR_v': # Rain number density
            d = file_instance.get_variable(['QNR','RHO'],**options)
            derived_var=d['QNR']*d['RHO']
            derived_var.name='QNR_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density'
            
        elif varname == 'QNS_v': # Snow number density
            d = file_instance.get_variable(['QNS','RHO'],**options)
            derived_var=d['QNS']*d['RHO']
            derived_var.name='QNS_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Snow number density'
            
        elif varname == 'QNG_v': # Graupel number density
            d = file_instance.get_variable(['QNG','RHO'],**options)
            derived_var=d['QNG']*d['RHO']
            derived_var.name='QNG_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Graupel number density'
            
        elif varname == 'QNC_v': # Cloud number density
            d = file_instance.get_variable(['QNC','RHO'],**options)
            derived_var=d['QNC']*d['RHO']
            derived_var.name='QNC_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density' 
            
        elif varname == 'QNI_v': # Ice cloud particles number density
            d = file_instance.get_variable(['QNI','RHO'],**options)
            derived_var=d['QNI']*d['RHO']
            derived_var.name='QNI_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Ice crystals number density'
            
        elif varname == 'QNH_v': # Hail number density
            d = file_instance.get_variable(['QNH','RHO'],**options)
            derived_var=d['QNH']*d['RHO']
            derived_var.name='QNH_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density'    
            
        elif varname == 'LWC': # LIQUID WATER CONTENT
            d = file_instance.get_variable(['QC','QR'],**options)
            derived_var=d['QC']+d['QR']
            derived_var=derived_var*100000
            derived_var.name='LWC'
            derived_var.attributes['units']='mg/kg'
            derived_var.attributes['long_name']='Liquid water content'
            
        elif varname == 'IWC': # ICE WATER CONTENT
            d = file_instance.get_variable(['QG','QS','QI'],**options)
            derived_var=d['QG']+d['QS']+d['QI']
            derived_var=derived_var*100000
            derived_var.name='IWC'
            derived_var.attributes['units']='mg/kg'
            derived_var.attributes['long_name']='Ice water content'
            
        elif varname == 'TWC': # TOTAL WATER CONTENT 
            d = file_instance.get_variable(['QG','QS','QI','QC','QV','QR'],**options)
            derived_var=d['QG']+d['QS']+d['QI']+d['QC']+d['QV']+d['QR']
            derived_var=derived_var*100000
            derived_var.name='TWC'
            derived_var.attributes['long_name']='Total water content'
            derived_var.attributes['units']='mg/kg'         
            
        elif varname == 'RHO': # AIR DENSITY
            d = file_instance.get_variable(['P','T','QV','QR','QC','QI','QS','QG'],**options)
            derived_var=d['P']/(d['T']*COSMO_R_D*((d['QV']*COSMO_RVD_M_O\
            -d['QR']-d['QC']-d['QI']-d['QS']-d['QG'])+1.0))
            derived_var.name='RHO'
            derived_var.attributes['long_name']='Air density'
            derived_var.attributes['units']='kg/m3'     
            
        elif varname == 'Pw': # Vapor pressure
            d = file_instance.get_variable(['P','QV'],**options)
            derived_var=(d['P']*d['QV'])/(d['QV']*(1-0.6357)+0.6357)
            derived_var.attributes['long_name']='Vapor pressure'
            derived_var.attributes['units']='Pa'            
            
        elif varname == 'RELHUM': # Vapor pressure
            d = file_instance.get_variable(['Pw','T'],**options)
            esat = 610.78*np.exp(17.2693882*(d['T'].data-273.16)/(d['T'].data-35.86)) # TODO
            derived_var=d['Pw']/esat*100
            derived_var.attributes['long_name']='Relative humidity'
            derived_var.attributes['units']='%'          
            
        elif varname == 'N': # Refractivity
            d = file_instance.get_variable(['T','Pw','P'],**options)
            derived_var=(77.6/d['T'])*(0.01*d['P']+4810*(0.01*d['Pw'])/d['T'])
            derived_var.attributes['long_name']='Refractivity'
            derived_var.attributes['units']='-'     
            
        else:
            raise ValueError('Could not compute derived variable, please specify a valid variable name')
    except:
        raise ValueError('Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.')        
    return derived_var 