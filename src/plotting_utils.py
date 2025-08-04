import plotly.graph_objs as go
import plotly.io as pio
import webbrowser
import tempfile
import os

def plot_training_metrics(metrics_dict, display_mode='inline', html_path=None):
    """
    Plot training metrics with a dropdown to select which metric to display.

    This function visualizes training metrics (such as loss, accuracy, etc.) over iterations.
    The input is a dictionary with at least an 'iteration' key (list of iteration numbers) and
    one or more other keys representing metric names (each mapping to a list of values).
    The plot uses Plotly and provides a dropdown to select which metric to display on the y-axis.

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
        metrics = {
            'iteration': [1, 2, 3, 4],
            'loss': [0.9, 0.7, 0.5, 0.3],
            'accuracy': [0.5, 0.6, 0.7, 0.8]
        }
        plot_training_metrics(metrics, display_mode='inline')
    """
    if 'iteration' not in metrics_dict:
        raise ValueError("metrics_dict must contain an 'iteration' key.")

    x = metrics_dict['iteration']
    metric_keys = [k for k in metrics_dict.keys() if k != 'iteration']

    if not metric_keys:
        raise ValueError("metrics_dict must contain at least one metric key besides 'iteration'.")

    # Create traces for each metric, only first is visible
    traces = []
    for i, key in enumerate(metric_keys):
        traces.append(go.Scatter(
            x=x,
            y=metrics_dict[key],
            mode='lines+markers',
            name=key,
            visible=(i == 0)
        ))

    # Dropdown buttons for each metric
    buttons = []
    for i, key in enumerate(metric_keys):
        visible = [False] * len(metric_keys)
        visible[i] = True
        buttons.append(dict(
            label=key,
            method='update',
            args=[{'visible': visible},
                  {'yaxis': {'title': key}}]
        ))

    layout = go.Layout(
        title='Training Metrics',
        xaxis=dict(title='Iteration'),
        yaxis=dict(title=metric_keys[0]),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.15,
            y=1.15
        )]
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