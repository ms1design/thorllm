#!/bin/bash
# lib/completion.sh — bash/zsh TAB completion for thorllm
# Install:
#   bash: source /path/to/thorllm/lib/completion.sh
#   zsh:  autoload -Uz compinit && compinit
#         source /path/to/thorllm/lib/completion.sh
# =============================================================================

# ── Bash completion ───────────────────────────────────────────────────────────
_thorllm_complete_bash() {
    local cur prev words cword
    _init_completion 2>/dev/null || {
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword="${COMP_CWORD}"
    }

    local commands="setup install start stop restart status logs kill model version config uninstall"
    local model_cmds="list add switch select show edit"
    local global_opts="--help --no-color --build-path -h"

    case "${prev}" in
        thorllm)
            COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- "${cur}") )
            return 0
            ;;
        start)
            COMPREPLY=( $(compgen -W "--port -p" -- "${cur}") )
            return 0
            ;;
        --port|-p)
            COMPREPLY=( $(compgen -W "8000 8080 9000 11434" -- "${cur}") )
            return 0
            ;;
        install)
            COMPREPLY=( $(compgen -W "--update" -- "${cur}") )
            return 0
            ;;
        logs)
            COMPREPLY=( $(compgen -W "-f" -- "${cur}") )
            return 0
            ;;
        model)
            COMPREPLY=( $(compgen -W "${model_cmds}" -- "${cur}") )
            return 0
            ;;
        add|switch|show|edit)
            # Try to complete from known model configs
            local models_dir="${BUILD_PATH:-${HOME}/thorllm}/models"
            if [[ -d "${models_dir}" ]]; then
                local models
                models=$(find "${models_dir}" -name "*.yaml" ! -path "*/example/*" 2>/dev/null \
                    | sed "s|${models_dir}/||;s|\.yaml$||")
                COMPREPLY=( $(compgen -W "${models}" -- "${cur}") )
            fi
            return 0
            ;;
        --build-path)
            COMPREPLY=( $(compgen -d -- "${cur}") )
            return 0
            ;;
    esac

    # Second-level: thorllm model <subcmd>
    if [[ "${cword}" -ge 2 && "${words[1]}" == "model" ]]; then
        case "${words[2]}" in
            add|switch|show|edit)
                local models_dir="${BUILD_PATH:-${HOME}/thorllm}/models"
                if [[ -d "${models_dir}" ]]; then
                    local models
                    models=$(find "${models_dir}" -name "*.yaml" ! -path "*/example/*" 2>/dev/null \
                        | sed "s|${models_dir}/||;s|\.yaml$||")
                    COMPREPLY=( $(compgen -W "${models}" -- "${cur}") )
                fi
                return 0
                ;;
        esac
    fi

    COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- "${cur}") )
}

# ── Zsh completion ────────────────────────────────────────────────────────────
_thorllm_complete_zsh() {
    local -a commands model_cmds

    commands=(
        'setup:Interactive TUI setup wizard'
        'install:Run full installation'
        'start:Start the vLLM service'
        'stop:Stop the vLLM service'
        'restart:Restart the vLLM service'
        'status:Show service status'
        'logs:Show service logs'
        'kill:Force-kill vLLM and free GPU memory'
        'model:Model management'
        'version:Show installed versions'
        'config:Show current configuration'
        'uninstall:Remove thorllm'
    )
    model_cmds=(
        'list:List available model configs'
        'add:Create a new model config'
        'switch:Switch active model'
        'select:Interactive model selection'
        'show:Print model config YAML'
        'edit:Edit model config in $EDITOR'
    )

    local state
    _arguments \
        '(-h --help)'{-h,--help}'[Show help]' \
        '--no-color[Disable colour output]' \
        '--build-path[Override BUILD_PATH]:directory:_files -/' \
        '1:command:->command' \
        '*::args:->args' \
    && return 0

    case "${state}" in
        command)
            _describe 'thorllm commands' commands
            ;;
        args)
            case "${words[1]}" in
                start)
                    _arguments \
                        '(-p --port)'{-p,--port}'[Port to listen on]:port:(8000 8080 9000)' \
                    ;;
                install)
                    _arguments '--update[Update vLLM to latest]'
                    ;;
                logs)
                    _arguments '-f[Follow log output]'
                    ;;
                model)
                    if (( CURRENT == 2 )); then
                        _describe 'model subcommands' model_cmds
                    else
                        # Complete model names
                        local models_dir="${BUILD_PATH:-${HOME}/thorllm}/models"
                        if [[ -d "${models_dir}" ]]; then
                            local models
                            models=( $(find "${models_dir}" -name "*.yaml" ! -path "*/example/*" 2>/dev/null \
                                | sed "s|${models_dir}/||;s|\.yaml$||") )
                            _multi_parts / models
                        fi
                    fi
                    ;;
            esac
            ;;
    esac
}

# ── Register completion ───────────────────────────────────────────────────────
if [[ -n "${ZSH_VERSION:-}" ]]; then
    compdef _thorllm_complete_zsh thorllm
elif [[ -n "${BASH_VERSION:-}" ]]; then
    complete -F _thorllm_complete_bash thorllm
fi

# ── Print install hint ────────────────────────────────────────────────────────
thorllm_completion_install_hint() {
    local script_path
    script_path=$(realpath "${BASH_SOURCE[0]}")
    echo ""
    echo "To enable TAB completion permanently, add to your shell rc:"
    echo ""
    echo "  # bash (~/.bashrc):"
    echo "  source ${script_path}"
    echo ""
    echo "  # zsh (~/.zshrc):"
    echo "  autoload -Uz compinit && compinit"
    echo "  source ${script_path}"
    echo ""
}
