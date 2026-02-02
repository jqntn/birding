using MauiAppSample.Models;
using MauiAppSample.PageModels;

namespace MauiAppSample.Pages
{
    public partial class MainPage : ContentPage
    {
        public MainPage(MainPageModel model)
        {
            InitializeComponent();
            BindingContext = model;
        }
    }
}