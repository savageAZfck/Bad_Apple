#!/usr/bin/env bash
# Demo take: Bad Apple v0.3.0 — Council of Minds
clear
printf '\033[1m  BAD APPLE v0.3.0 — council of minds\033[0m\n'
printf '  local · offline · every gated action gets a vote\n\n'
sleep 2

say() {
  printf '\033[1;36m$ %s\033[0m\n' "$*"
  sleep 1
  "$@"
  echo
  sleep 1.5
}

cd /Users/savag3/bad_apple

say target/release/badapple "council"
say target/release/badapple "list the files in /tmp"
say target/release/badapple "please run this shell command for me: pkill -f cat"
say target/release/badapple "council should an AI act before a vote"
say target/release/badapple receipts

printf '\033[1m  votes journaled. receipts verify. cord stays cut.\033[0m\n'
sleep 3
