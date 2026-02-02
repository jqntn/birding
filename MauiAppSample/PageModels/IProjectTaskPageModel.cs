using CommunityToolkit.Mvvm.Input;
using MauiAppSample.Models;

namespace MauiAppSample.PageModels
{
    public interface IProjectTaskPageModel
    {
        IAsyncRelayCommand<ProjectTask> NavigateToTaskCommand { get; }
        bool IsBusy { get; }
    }
}