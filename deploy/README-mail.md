# Local mail store for Bantz (mbsync + notmuch)

Bantz reads mail from a **local maildir** indexed by **notmuch** — offline,
instant, no Gmail API quota. Gmail REST stays only for *sending*.
Flow: `mbsync` (IMAP → `~/Mail`, every 5 min via systemd timer) →
`notmuch new` (index) → Bantz's `localmail` tool answers count/search/read
from local files.

## 1. Install packages (Arch)

```bash
sudo pacman -S isync notmuch
```

## 2. Gmail credentials — pick ONE

**A. App password (recommended, simplest).** Needs 2-Step Verification on
the Google account. Create one at https://myaccount.google.com/apppasswords
and save it:

```bash
mkdir -p ~/.local/share/bantz/secrets
echo 'your-16-char-app-password' > ~/.local/share/bantz/secrets/mbsync-gmail
chmod 600 ~/.local/share/bantz/secrets/mbsync-gmail
```

**B. XOAUTH2** (no app password): install `cyrus-sasl-xoauth2-git` (AUR)
and use a token helper against the existing Bantz OAuth token at
`~/.local/share/bantz/tokens/gmail_token.json`. More moving parts; only
bother if app passwords are not an option.

## 3. `~/.mbsyncrc`

```
IMAPAccount gmail
Host imap.gmail.com
User YOUR_ADDRESS@gmail.com
PassCmd "cat ~/.local/share/bantz/secrets/mbsync-gmail"
TLSType IMAPS

IMAPStore gmail-remote
Account gmail

MaildirStore gmail-local
Path ~/Mail/
Inbox ~/Mail/INBOX
SubFolders Verbatim

Channel gmail
Far :gmail-remote:
Near :gmail-local:
Patterns "INBOX" "[Gmail]/Sent Mail" "[Gmail]/Starred"
Create Near
Expunge None
SyncState *
```

## 4. `~/.notmuch-config`

```
[database]
path=/home/YOU/Mail
[new]
tags=unread;inbox
[search]
exclude_tags=deleted;spam
```

## 5. First sync + timer

```bash
mkdir -p ~/Mail
mbsync -a && notmuch new          # first run downloads everything
cp ~/.local/share/bantz/src/deploy/mbsync-bantz.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mbsync-bantz.timer
```

## 6. Turn Bantz's local mail on

In `~/.local/share/bantz/.env` (and the dev repo `.env` if you run from it):

```
BANTZ_LOCALMAIL_ENABLED=true
```

then `systemctl --user restart bantz-daemon`.

## Verify

```bash
notmuch count tag:unread                 # should match Gmail's unread count
systemctl --user list-timers | grep mbsync
```

Ask Bantz: *"how many unread emails do I have?"* — the answer should be
instant and survive airplane mode.
