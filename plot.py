import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For formatting the y-axis

# Updated training and validation loss values from your new log
train_losses = [0.1580, 0.0493, 0.0194, 0.0107, 0.0071]
val_losses = [0.1033, 0.0978, 0.1077, 0.1129, 0.1173]

# Define epoch numbers for the x-axis (1, 2, 3, 4, 5)
epochs = [1, 2, 3, 4, 5]

# Plotting commands, keeping structure from previous version
plt.ioff()

# Plot data against the actual epoch numbers
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch') # Duplicated as in your original function
plt.ylabel('Loss') # Duplicated as in your original function
plt.title('Training vs. Validation Loss')

# Set x-axis ticks to be precisely the integer epoch numbers
plt.xticks(epochs)

# Format y-axis tick labels to show one decimal place (e.g., 0.1, 0.2)
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

plt.savefig("loss_plot.jpg")

print("Plot saved as loss_plot.jpg in the current working directory.")
print("X-axis (Epochs) should now show integers (1, 2, 3, 4, 5).")
print("Y-axis (Loss) should now be formatted to one decimal place (e.g., 0.1).")
print("The data used for this plot is from your latest training log.")