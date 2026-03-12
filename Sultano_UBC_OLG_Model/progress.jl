module Progress

export ProgressBar, update!, finish!

using Printf

"""
    ProgressBar

Simple progress bar for long-running computations.
Uses only standard library (Printf).

Fields:
- `label`: Description of what's being computed
- `total`: Total number of iterations expected
- `current`: Current iteration (0-based)
- `width`: Width of progress bar in characters (default 50)
- `show_percent`: Whether to show percentage (default true)
- `show_eta`: Whether to show estimated time remaining (default true)
- `start_time`: Timestamp when progress bar was created
- `last_update`: Timestamp of last update (for rate estimation)
"""
mutable struct ProgressBar
    label::String
    total::Int
    current::Int
    width::Int
    show_percent::Bool
    show_eta::Bool
    start_time::Float64
    last_update::Float64
    last_current::Int
end

"""
    ProgressBar(label, total; width=50, show_percent=true, show_eta=true)

Create a new progress bar.

# Example
```julia
pb = ProgressBar("Calibrating", 50)
for i in 1:50
    # ... do work ...
    update!(pb, i)
end
finish!(pb)
```
"""
function ProgressBar(label::String, total::Int; width::Int=50, show_percent::Bool=true, show_eta::Bool=true)
    pb = ProgressBar(label, total, 0, width, show_percent, show_eta, 
                     time(), time(), 0)
    print_progress(pb)
    return pb
end

"""
    print_progress(pb)

Internal function to print the progress bar.
"""
function print_progress(pb::ProgressBar)
    # Clear line and move cursor to beginning
    print("\r\033[K")
    
    # Label
    print(pb.label, ": ")
    
    # Progress bar
    if pb.total > 0
        filled = round(Int, pb.width * pb.current / pb.total)
        filled = clamp(filled, 0, pb.width)
        filled_chars = filled > 0 ? "█"^filled : ""
        empty_chars = "░"^(pb.width - filled)
        bar = filled_chars * empty_chars
        print("[", bar, "]")
        
        # Percentage
        if pb.show_percent
            pct = 100.0 * pb.current / pb.total
            @printf(" %.1f%%", pct)
        end
        
        # ETA
        if pb.show_eta && pb.current > 0
            elapsed = time() - pb.start_time
            if pb.current > pb.last_current && elapsed > 0.1
                rate = pb.current / elapsed
                remaining = max(0, pb.total - pb.current)
                eta_seconds = remaining / rate
                
                if eta_seconds < 60
                    @printf(" ETA: %.0fs", eta_seconds)
                elseif eta_seconds < 3600
                    @printf(" ETA: %.1fm", eta_seconds / 60)
                else
                    @printf(" ETA: %.1fh", eta_seconds / 3600)
                end
            end
        end
    else
        # Indeterminate progress
        print("[", "░"^pb.width, "]")
    end
    
    flush(stdout)
end

"""
    update!(pb, current)

Update the progress bar to show `current` iterations completed.
"""
function update!(pb::ProgressBar, current::Int)
    pb.last_current = pb.current
    pb.current = clamp(current, 0, pb.total)
    pb.last_update = time()
    print_progress(pb)
end

"""
    finish!(pb)

Mark the progress bar as complete and print final status.
"""
function finish!(pb::ProgressBar)
    pb.current = pb.total
    print_progress(pb)
    elapsed = time() - pb.start_time
    if elapsed < 60
        @printf(" (%.2fs)\n", elapsed)
    elseif elapsed < 3600
        @printf(" (%.1fm)\n", elapsed / 60)
    else
        @printf(" (%.1fh)\n", elapsed / 3600)
    end
    flush(stdout)
end

"""
    finish!(pb, message)

Mark the progress bar as complete with a custom message.
"""
function finish!(pb::ProgressBar, message::String)
    pb.current = pb.total
    print_progress(pb)
    elapsed = time() - pb.start_time
    if elapsed < 60
        @printf(" %s (%.2fs)\n", message, elapsed)
    elseif elapsed < 3600
        @printf(" %s (%.1fm)\n", message, elapsed / 60)
    else
        @printf(" %s (%.1fh)\n", message, elapsed / 3600)
    end
    flush(stdout)
end

end
