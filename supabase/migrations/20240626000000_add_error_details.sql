-- Add error_details column to process_logs table
ALTER TABLE IF EXISTS process_logs 
ADD COLUMN IF NOT EXISTS error_details jsonb;

-- Comment on the new column
COMMENT ON COLUMN process_logs.error_details IS 'Detailed error information in JSON format, including context data from when the error occurred'; 