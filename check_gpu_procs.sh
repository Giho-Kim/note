#!/bin/bash
# nvidia-smi에 잡히는 각 프로세스(PID)가 실제로 어떤 커맨드로 실행됐는지 보여준다.

echo "PID       GPU   MEM       CMD"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=',' read -r pid mem; do
    pid=$(echo "$pid" | xargs)
    mem=$(echo "$mem" | xargs)
    cmd=$(ps -p "$pid" -o args= 2>/dev/null)
    if [ -z "$cmd" ]; then
        cmd="(프로세스 정보 없음, 다른 컨테이너/네임스페이스일 수 있음)"
    fi
    printf "%-9s %-8s %s\n" "$pid" "${mem}MiB" "$cmd"
done
