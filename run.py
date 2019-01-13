from ezaml import EzAml

ezaml = EzAml()
ezaml.train(
    script_params={'--data-folder': './storage'})