# For medi0000 series

for ($i = 0; $i -lt 7; $i++)
{
    # python -m pyrun --mode ("hello"+$i.ToString())
    python -m projects.ezmed.main --mode e2e --label ("medi0000-10" + $i.ToString())
    python -m projects.ezmed.main --mode e2e --label ("medi0000-20" + $i.ToString()) --C 0.4    
}
