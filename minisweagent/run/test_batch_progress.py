from extra.utils.batch_progress import RunBatchProgressManager

# Initialize the progress manager with 5 instances
progress_manager = RunBatchProgressManager(num_instances=5)

# Simulate starting and ending instances
for i in range(5):
    progress_manager.on_instance_start(f"instance_{i}")
    progress_manager.update_instance_status(f"instance_{i}", f"Processing {i}")
    progress_manager.on_instance_end(f"instance_{i}", "Success")

# Print the final report
progress_manager.print_report()
