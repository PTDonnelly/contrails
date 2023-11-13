import numpy as np
from pytmatrix.tmatrix import Scatterer
import pytmatrix.scatter as scatter
from pytmatrix.psd import PSDIntegrator
from pytmatrix.psd import GammaPSD
# from pytmatrix.tmatrix_aux import geom_horiz_back
import snoop

# @snoop
def main():
    # Define the lognormal particle size distribution
    # D0 = 5  # Example median diameter
    # Nw = 1  # Example total number concentration
    # mu = 4   # Example shape parameter
    # psd_function = GammaPSD(D0=D0, Nw=Nw, mu=mu)

    psd_function = GammaPSD()

    # Initialise integrator
    psd_integrator = PSDIntegrator(D_max=psd_function.D_max)
    psd_integrator.num_points = 10
    
    wavelengths = [7]
    for wl in wavelengths:
        # Initialise scatterer
        scatterer = Scatterer(wavelength=wl, m=1.33 + 0.01j, psd_integrator=psd_integrator)
        scatterer.psd = psd_function
        
        # Calculate scattering matrix
        psd_integrator.init_scatter_table(tm=scatterer, angular_integration=True, verbose=False)

        # Get scattering properties
        ext_xsect = scatter.ext_xsect(scatterer)
        sca_xsect = scatter.sca_xsect(scatterer) # Not used for 4A/OP ice
        abs_xsect = np.subtract(ext_xsect, sca_xsect) # Not used for 4A/OP ice
        ssa = scatter.ssa(scatterer)
        asym = scatter.asym(scatterer)
        print(ext_xsect, sca_xsect, abs_xsect, ssa, asym)

    # # Save scattering matrix
    # psd_integrator.save_scatter_table("scattering_matrix_10.pkl", description="test at 10 effective radius")

    # # Load scattering matrix
    # load_time, description = psd_integrator.load_scatter_table("scattering_matrix_10.pkl")
    # print(f"Table loaded from {load_time}: {description}")

if __name__ == "__main__":
    main()