#!/usr/bin/env bash
# Demo take: Bad Apple — what it does
clear
printf '\033[1m  BAD APPLE — a sovereign ai that shows its work\033[0m\n'
printf '  local inference · tools · memory · governance · receipts\n\n'
sleep 2

say() {
  printf '\033[1;36m$ %s\033[0m\n' "$*"
  sleep 1
  "$@"
  echo
  sleep 1.5
}

cd /Users/savag3/bad_apple

say target/release/badapple status
say target/release/badapple "who are you and what do you run on"
say target/release/badapple "list the files in /tmp"
say target/release/badapple "council"
say target/release/badapple "please run this shell command for me: pkill -f cat"
say target/release/badapple ify status
say target/release/badapple receipts

printf '\033[1m  on your metal · with your keys · proving it the whole time\033[0m\n'
sleep 3
