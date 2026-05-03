#!/usr/bin/env fish

function usage
    echo "Usage:"
    echo "  fish run_kdenlive_rocm_container.fish [--image IMAGE] build"
    echo "  fish run_kdenlive_rocm_container.fish [--image IMAGE] kdenlive-mask [kdenlive_face_mask args...]"
    echo ""
    echo "Options:"
    echo "  --image IMAGE  Override container image name (default: kdenlive-face-tracking-rocm)"
    echo "                 You can point this to an existing image built in another project."
end

if not command -q podman
    echo "podman is required for the ROCm container workflow" >&2
    exit 1
end

set -l script_dir (cd (dirname (status --current-filename)); pwd)
set -l repo_root "$script_dir"
set -l image_name kdenlive-face-tracking-rocm
set -l host_tool_home "$HOME/.cache/kdenlive-face-tracking-home"
set -l container_home /kdenlive-tool-home
set -l hsa_override_gfx_version 11.0.0

if set -q HSA_OVERRIDE_GFX_VERSION
    set hsa_override_gfx_version $HSA_OVERRIDE_GFX_VERSION
end

if test (count $argv) -lt 1
    usage
    exit 1
end

set -l command $argv[1]
set -l command_args $argv[2..-1]

if test "$command" = "--image"
    if test (count $argv) -lt 3
        echo "--image requires a value" >&2
        usage
        exit 1
    end
    set image_name $argv[2]
    set command $argv[3]
    set command_args $argv[4..-1]
end

switch $command
    case build
        set -l build_context (mktemp -d)
        if test $status -ne 0
            echo "failed to create temporary build context" >&2
            exit 1
        end

        podman build \
            --file "$repo_root/Containerfile.rocm" \
            --tag "$image_name" \
            "$build_context"

        set -l build_status $status
        rm -rf "$build_context"
        exit $build_status

    case kdenlive-mask
        mkdir -p \
            "$HOME/.cache/huggingface" \
            "$HOME/.insightface" \
            "$host_tool_home/.cache/matplotlib" \
            "$host_tool_home/.config/miopen"

        set -l mask_args \
            --provider-mode rocm \
            $command_args

        podman run --rm \
            --interactive \
            --name kdenlive-face-tracking-rocm \
            --replace \
            --userns keep-id \
            --group-add keep-groups \
            --device /dev/kfd \
            --device /dev/dri \
            --env HOME="$container_home" \
            --env HSA_OVERRIDE_GFX_VERSION="$hsa_override_gfx_version" \
            --env MPLCONFIGDIR="$container_home/.cache/matplotlib" \
            --env XDG_CACHE_HOME="$container_home/.cache" \
            --env XDG_CONFIG_HOME="$container_home/.config" \
            --env PYTHONDONTWRITEBYTECODE=1 \
            --volume "$repo_root:/workspace:ro" \
            --volume "$HOME:$HOME" \
            --volume "/tmp:/tmp" \
            --volume "$host_tool_home:$container_home" \
            --volume "$HOME/.insightface:$container_home/.insightface" \
            --volume "$HOME/.cache/huggingface:$container_home/.cache/huggingface" \
            --workdir "$PWD" \
            --entrypoint python \
            "$image_name" \
            /workspace/kdenlive_face_mask.py \
            $mask_args

    case '*'
        usage
        exit 1
end
