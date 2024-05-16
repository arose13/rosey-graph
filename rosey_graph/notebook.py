def auto_display(obj):
    """
    Switch between display or print based on whether we are in a notebook or not.
    """
    if 'get_ipython' in globals():
        from IPython.display import display
        display(obj)
    else:
        print(obj)
