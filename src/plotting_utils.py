import plotly.graph_objs as go
import plotly.io as pio
import webbrowser
import tempfile
import os
import numpy as np

def plot_training_metrics(metrics_dict, display_mode='inline', html_path=None):
    """
    Plot training metrics with a dropdown to select which metric to display.

    This function visualizes training metrics (such as loss, accuracy, etc.) over iterations.
    For scalar metrics, it shows line plots. For vector metrics (params, velocity, grads),
    it creates hexbin plots showing the distribution of values across iterations.

    Args:
        metrics_dict (dict):
            Dictionary containing training metrics. Must have:
                - 'iteration': list or array of iteration numbers (x-axis)
                - Other keys: each is a metric name mapping to a list/array of values (y-axis)
        display_mode (str, optional):
            How to display the plot. Options:
                - 'inline': Display inline (e.g., in a Jupyter notebook)
                - 'tab': Open the plot in a new browser tab
                - 'file': Save the plot as an HTML file (requires html_path)
            Default is 'inline'.
        html_path (str, optional):
            Path to save the HTML file if display_mode is 'file'.
            Required if display_mode is 'file'.

    Raises:
        ValueError: If 'iteration' key is missing, or no metric keys are present,
                    or if html_path is not provided when display_mode is 'file'.

    Example:
        # From SGD class
        sgd = SGDMomentum()
        # ... after training ...
        plot_training_metrics(sgd.history, display_mode='inline')
    """
    if 'iteration' not in metrics_dict:
        raise ValueError("metrics_dict must contain an 'iteration' key.")

    # Convert to lists to ensure Plotly compatibility
    x = list(metrics_dict['iteration'])
    
    # Separate scalar and vector metrics
    scalar_keys = []
    vector_keys = []
    
    for k in metrics_dict.keys():
        if k == 'iteration':
            continue
        
        data = metrics_dict[k]
        if len(data) > 0:
            # Check if this is a list of arrays/vectors
            first_item = data[0]
            if hasattr(first_item, '__len__') and not isinstance(first_item, (str, int, float)):
                vector_keys.append(k)
            else:
                scalar_keys.append(k)

    all_keys = scalar_keys + vector_keys
    if not all_keys:
        raise ValueError("metrics_dict must contain at least one plottable metric key besides 'iteration'.")

    # Create traces for each metric
    traces = []
    
    # Add scalar metrics as line plots
    for i, key in enumerate(scalar_keys):
        y_data = metrics_dict[key]
        # Convert numpy arrays to lists for Plotly compatibility
        if hasattr(y_data, 'tolist'):
            y_data = y_data.tolist()
        elif hasattr(y_data, '__iter__'):
            y_data = [float(val) if hasattr(val, 'item') else val for val in y_data]
        
        traces.append(go.Scatter(
            x=x,
            y=y_data,
            mode='lines+markers',
            name=key,
            visible=True if i == 0 else False
        ))
    
    # Add vector metrics as hexbin plots
    for i, key in enumerate(vector_keys):
        vector_data = metrics_dict[key]
        
        # Flatten all vectors and create corresponding iteration indices
        all_values = []
        all_iterations = []
        
        for iter_idx, vec in enumerate(vector_data):
            if hasattr(vec, 'flatten'):
                flat_vec = vec.flatten()
            else:
                flat_vec = np.array(vec).flatten()
            
            all_values.extend(flat_vec)
            all_iterations.extend([x[iter_idx]] * len(flat_vec))
        
        traces.append(go.Histogram2d(
            x=all_iterations,
            y=all_values,
            name=f'{key} (hexbin)',
            colorscale='RdBu',
            zmid=0,
            xbins=dict(start=min(x)-0.5, end=max(x)+0.5, size=1),
            visible=True if len(scalar_keys) == 0 and i == 0 else False
        ))

    # Dropdown buttons for each metric
    buttons = []
    trace_idx = 0
    
    for key in scalar_keys:
        visible = [False] * len(traces)
        visible[trace_idx] = True
        buttons.append(dict(
            label=key,
            method='update',
            args=[{'visible': visible},
                  {'yaxis': {'title': key}}]
        ))
        trace_idx += 1
    
    for key in vector_keys:
        visible = [False] * len(traces)
        visible[trace_idx] = True
        buttons.append(dict(
            label=f'{key} (hexbin)',
            method='update',
            args=[{'visible': visible},
                  {'yaxis': {'title': f'{key} values'}}]
        ))
        trace_idx += 1

    layout = go.Layout(
        title='Training Metrics',
        xaxis=dict(title='Iteration'),
        yaxis=dict(title=all_keys[0] if all_keys else 'Value'),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.02,
            xanchor="left",
            y=1.02,
            yanchor="top"
        )] if len(buttons) > 0 else []
    )

    fig = go.Figure(data=traces, layout=layout)

    if display_mode == 'inline':
        # For Jupyter notebooks
        pio.show(fig)
    elif display_mode == 'tab':
        # Save to temp file and open in browser
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            fig.write_html(tmpfile.name)
            webbrowser.open_new_tab('file://' + os.path.abspath(tmpfile.name))
    elif display_mode == 'file':
        if not html_path:
            raise ValueError("html_path must be specified when display_mode is 'file'.")
        fig.write_html(html_path)
        print(f"Plot saved to {html_path}")
    else:
        raise ValueError("display_mode must be one of: 'inline', 'tab', 'file'.")