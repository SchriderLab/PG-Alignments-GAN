import stdpopsim
import tskit
import sys

n_reps = sys.argv[1]

for i in range(int(n_reps)):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_2T12")
    contig = species.get_contig("chr22", length_multiplier=0.001)
    samples = model.get_samples(32,32)
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples)

    if i == 0:
        with open("output.ms", "w") as ms_file:
            tskit.write_ms(ts,ms_file)
    else:
        with open("output.ms", "a") as ms_file:
            tskit.write_ms(ts,ms_file)