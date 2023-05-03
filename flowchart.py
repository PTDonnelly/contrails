from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank

with Diagram("QUISAIT Algorithm", show=False):
    input_data = Blank("Input Data")
    reference_atmosphere = Blank("Reference Atmosphere (IASI L2)")
    scattering = Blank("Scattering Properties (Water Ice, T-Matrix)")
    forward_model = Blank("Forward Model")
    retrieval = Blank("Retrieval")
    error_analysis = Blank("Error Analysis")
    output_data = Blank("Output Data")

    with Cluster("Radiative Transfer"):
        rt_input = Blank("RT Input")
        rt_module = Blank("RT Module")
        rt_output = Blank("RT Output")

    with Cluster("Retrieval"):
        retrieval_input = Blank("Retrieval Input")
        retrieval_module = Blank("Retrieval Module")
        retrieval_output = Blank("Retrieval Output")

    input_data >> forward_model
    reference_atmosphere >> forward_model
    scattering >> forward_model
    forward_model >> rt_input
    
    rt_input >> rt_module >> rt_output
    rt_output >> retrieval_input
    retrieval_input >> retrieval_module >> retrieval_output
    retrieval_module >> forward_model
    retrieval_output >> error_analysis >> output_data