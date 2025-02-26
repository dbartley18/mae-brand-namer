#!/usr/bin/env python
"""Command-line interface for Mae Brand Namer."""

import asyncio
import click
import json
import sys

from mae_brand_namer.workflows.brand_naming import run_brand_naming_workflow
from mae_brand_namer.config.settings import settings


@click.group()
def cli():
    """Mae Brand Namer - A LangGraph-powered brand naming tool."""
    pass


@cli.command()
@click.argument("prompt", required=True)
@click.option("--output", "-o", type=click.Path(), help="Save output to a JSON file")
def run(prompt, output):
    """
    Run the brand naming workflow with the given prompt.
    
    PROMPT is the description of the brand/company for which to generate names.
    """
    click.echo(f"Starting brand naming workflow for: {prompt}")
    click.echo("This may take a few minutes to complete...")
    
    try:
        # Run the async workflow in the event loop
        result = asyncio.run(run_brand_naming_workflow(prompt))
        
        # Print a summary to the console
        click.echo("\n✓ Workflow completed successfully!")
        click.echo(f"  Run ID: {result.get('run_id')}")
        click.echo(f"  Generated {len(result.get('generated_names', []))} brand name candidates")
        
        shortlisted = result.get('shortlisted_names', [])
        click.echo(f"  Shortlisted {len(shortlisted)} names")
        
        if shortlisted:
            click.echo("\nShortlisted brand names:")
            for name in shortlisted:
                click.echo(f"  • {name}")
        
        # Save full results to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"\nFull results saved to: {output}")
        
        # Show report URL if available
        if result.get('report_url'):
            click.echo(f"\nDetailed report available at: {result.get('report_url')}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def config():
    """Show current configuration settings."""
    click.echo("Mae Brand Namer Configuration:")
    click.echo(f"  Version: {settings.version}")
    click.echo(f"  Environment: {settings.environment}")
    click.echo(f"  LangSmith Enabled: {settings.langsmith_enabled}")
    if settings.langsmith_enabled:
        click.echo(f"  LangSmith Project: {settings.langsmith_project}")
    click.echo(f"  Max Retries: {settings.max_retries}")


if __name__ == "__main__":
    cli() 