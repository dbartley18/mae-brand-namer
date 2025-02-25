-- Create process_logs table for tracking workflow execution
create table if not exists process_logs (
    id uuid default uuid_generate_v4() primary key,
    run_id text not null,
    agent_type text not null,
    task_name text not null,
    status text not null check (status in ('in_progress', 'completed', 'error')),
    start_time timestamp with time zone not null,
    end_time timestamp with time zone,
    duration_sec integer,
    input_size_kb integer,
    output_size_kb integer,
    error_message text,
    retry_count integer default 0,
    last_retry_at timestamp with time zone,
    retry_status text check (retry_status in ('pending', 'exhausted', null)),
    last_updated timestamp with time zone not null,
    created_at timestamp with time zone default now() not null,
    
    -- Indexes for common queries
    constraint process_logs_run_task_unique unique (run_id, task_name)
);

-- Create index for querying by run_id
create index if not exists process_logs_run_id_idx on process_logs (run_id);

-- Create index for querying by task_name
create index if not exists process_logs_task_name_idx on process_logs (task_name);

-- Create index for querying by status
create index if not exists process_logs_status_idx on process_logs (status);

-- Enable Row Level Security
alter table process_logs enable row level security;

-- Create policy to allow all operations for authenticated users
create policy "Allow all operations for authenticated users"
  on process_logs
  for all
  to authenticated
  using (true); 