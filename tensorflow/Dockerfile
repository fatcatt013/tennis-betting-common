FROM tensorflow/tensorflow:latest-gpu

# Install Jupyter Notebook, Vim, and scikit-learn (assuming you need scikit-learn)
RUN pip install jupyter pandas scikit-learn && \
    apt-get update && \
    apt-get install -y vim

# Expose port for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]