Contents
--------
fly_streams.html
    The interactive page (the "GUI"). Self-contained: just open it in a browser,
    all data is embedded. Internet is only used to load the display font;
    everything works offline too.

fig1_fly_streams_300dpi.png
    The figure from the proposal at 300 dpi (7200 x 7956 px), suitable for print.

code/
    The scripts that build both, in pipeline order:
      fly_ann_prep.py         joins the public FlyWire annotations to the graph
      fly_streams.py          builds the (stream x depth) layout + numbers
      fly_cloud.py            per-neuron positions for the zoomed-in view
      fly_artifact_export.py  bundles everything into one JSON
      fly_streams_page.html   the page itself (HTML/JS/CSS template)
      build_fly_html.py       inlines the JSON into the final fly_streams.html
      fly_streams_proposal.py the static proposal figure (the PNG above)
    The scripts expect to run inside the repository (they read the connectome
    and the optimized ordering from it), so they are for reference; the HTML
    and PNG above are ready to use as-is.
