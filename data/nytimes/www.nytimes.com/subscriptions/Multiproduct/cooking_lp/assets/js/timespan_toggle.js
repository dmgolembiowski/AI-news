(function ($) {
    $(document).ready(function(){
        
        $('.payment_length .pay_btn.length1').on("click",function() {
            if(!$(this).hasClass("active")){
                $('.price2').hide().removeClass("active_cta"); 
                $('span.price1').css({display: "inline"});
                $('a.cta.price1').css({display: "inline-block"}).addClass("active_cta");
                $('.payment_length .pay_btn.active').find(".active_bg").animate({width: "0"}, 300);
                $('.payment_length .pay_btn.active').removeClass('active');
                $(this).find(".active_bg").animate({width: "100%"}, 300, function(){
                    $(".payment_length .pay_btn.lenght1").addClass('active');
                });   
            }
        });//payment time span click
            
        $('.payment_length .pay_btn.length2').on("click",function() {
            if(!$(this).hasClass("active")){
                $('.price1').hide().removeClass("active_cta"); 
                $('span.price2').css({display: "inline"});
                $('a.cta.price2').css({display: "inline-block"}).addClass("active_cta");
                $('.payment_length .pay_btn.active').find(".active_bg").animate({width: "0"}, 300);
                $('.payment_length .pay_btn.active').removeClass('active');
                $(this).find(".active_bg").animate({width: "100%"}, 300, function(){
                    $(".payment_length .pay_btn.lenght2").addClass('active');
                });  
            }  
        });//payment time span click
        
        $(".bundle_prod").hover(function(){
            var activeLink = $(this).find("a.cta.active_cta").attr("href"); 
            console.log(activeLink); 
        });
        
    });//END of Doc
})(jQuery);